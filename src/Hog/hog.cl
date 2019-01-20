#define SENS_BINS 18
#define BINS_PER_BLOCK 31

#define TRUNC 0.2f

#define HOG_WG_SZ_BIG 16
#define HOG_WG_SZ_BIG_LIN 256

#define HOG_IM_LOC_SZ 18
#define HOG_IM_LOC_SZ_LIN 324

#define HOG_WG_SZ_SMALL 4
#define HOG_WG_SZ_SMALL_LIN 16

inline void calcDerivsInl(
    __local const float* const restrict im,
    __global float* const restrict derivX,
    __global float* const restrict derivY,
    const int derivId)
{
    derivX[derivId] = im[1] - im[-1];
    derivY[derivId] = im[HOG_IM_LOC_SZ] - im[-HOG_IM_LOC_SZ];
}

__kernel void calcDerivs(
    __global const float* const restrict imGlob,
    __global float* const restrict derivsX,
    __global float* const restrict derivsY,
    __local float* const restrict imLoc,
    const int iterCnt,
    const int halfPad)
{
    const int2 wiId = (int2)(get_local_id(0), get_local_id(1));
    const int2 imGlobSz = (int2)(get_global_size(0), mul24((int)HOG_WG_SZ_BIG, iterCnt));
    const int derivsSzX = imGlobSz.x + 2 * halfPad;
    const int derivIdLoc = mad24(wiId.y + 1, HOG_IM_LOC_SZ, wiId.x + 1);
    const int imGlobIterStep = mul24((int)HOG_WG_SZ_BIG, imGlobSz.x);
    const int derivsIterStep = mul24((int)HOG_WG_SZ_BIG, derivsSzX);

    int derivsShift = mad24(wiId.y + halfPad, derivsSzX, (int)get_global_id(0) + halfPad);
    int srcIdLoc[2];
    int srcIdGlob[2];
    {
        const int wiIdLin = mad24(wiId.y, (int)HOG_WG_SZ_BIG, wiId.x);
        const int wgShiftX = mul24((int)get_group_id(0), (int)HOG_WG_SZ_BIG);
        #pragma unroll 2
        for (int i = 0; i < 2; ++i)
        {
            srcIdLoc[i] = (wiIdLin + mul24(i, (int)HOG_WG_SZ_BIG_LIN)) % HOG_IM_LOC_SZ_LIN;
            srcIdGlob[i] = mad24(srcIdLoc[i] / HOG_IM_LOC_SZ - 1, imGlobSz.x,
                clamp(srcIdLoc[i] % HOG_IM_LOC_SZ + wgShiftX - 1, 0, imGlobSz.x - 1));
        }
    }

    #pragma unroll 2
    for (int i = 0; i < 2; ++i)
    {
        imLoc[srcIdLoc[i]] = imGlob[srcIdGlob[i] + (srcIdGlob[i] >= 0 ? 0 : imGlobSz.x)];
        srcIdGlob[i] += imGlobIterStep;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    calcDerivsInl(imLoc + derivIdLoc, derivsX, derivsY, derivsShift);
    derivsShift += derivsIterStep;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int iter = 1; iter + 1 < iterCnt; ++iter, derivsShift += derivsIterStep)
    {
        #pragma unroll 2
        for (int i = 0; i < 2; ++i)
        {
            imLoc[srcIdLoc[i]] = imGlob[srcIdGlob[i]];
            srcIdGlob[i] += imGlobIterStep;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        calcDerivsInl(imLoc + derivIdLoc, derivsX, derivsY, derivsShift);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    imLoc[srcIdLoc[0]] = imGlob[srcIdGlob[0]];
    imLoc[srcIdLoc[1]] = imGlob[srcIdGlob[1] -
        (srcIdGlob[1] < mul24(imGlobSz.x, imGlobSz.y) ? 0 : imGlobSz.x)];
    barrier(CLK_LOCAL_MEM_FENCE);
    calcDerivsInl(imLoc + derivIdLoc, derivsX, derivsY, derivsShift);
}

__kernel void calcCellDesc(
    __global const float* restrict derivsXglob,
    __global const float* restrict derivsYglob,
    __global uint* const restrict cellDescGlob,
    __local float* const restrict derivsXloc,
    __local float* const restrict derivsYloc,
    __local uint* const restrict cellDescLoc,
    const int iterCnt,
    const int cellSz,
    const int2 halfPad)
{
    const int2 wiId = (int2)(get_local_id(0), get_local_id(1));
    const int derivsSzLoc = HOG_WG_SZ_BIG + cellSz;
    const int wiIdLin = mad24(wiId.y, (int)HOG_WG_SZ_BIG, wiId.x);
    const int imGlobSzX = get_global_size(0) + 2 * halfPad.x;
    const int derivsSzGlobX = imGlobSzX + cellSz;

    const int derivsPerIter = mul24((int)HOG_WG_SZ_BIG, derivsSzGlobX);
    int srcIdLoc[2];
    int srcIdGlob[2];
    {
        const int derivsSzLocLin = mul24(derivsSzLoc, derivsSzLoc);
        const int wgShiftX = mul24((int)get_group_id(0), (int)HOG_WG_SZ_BIG);
        #pragma unroll 2
        for (int i = 0; i < 2; ++i)
        {
            srcIdLoc[i] = mad24(i, (int)HOG_WG_SZ_BIG_LIN, wiIdLin) % derivsSzLocLin;
            srcIdGlob[i] = mad24(srcIdLoc[i] / derivsSzLoc + halfPad.y, derivsSzGlobX,
                srcIdLoc[i] % derivsSzLoc + wgShiftX + halfPad.x);
        }
    }

    const int2 cellCntLoc = HOG_WG_SZ_BIG / cellSz;
    const int2 cellIdLoc = wiId / cellSz;
    const int interpCellId = mul24(mad24(cellIdLoc.y, cellCntLoc.x, cellIdLoc.x), (int)SENS_BINS);

    int derivIdsLin[4];
    float interpCellWeights[4];
    {
        const int2 neighbId = cellSz - wiId % cellSz;
        #pragma unroll 4
        for (int i = 0; i < 4; ++i)
        {
            const int2 adjacent = mul24((int2)(i % 2, i / 2), cellSz);
            derivIdsLin[i] = mad24(wiId.y + adjacent.y, derivsSzLoc, wiId.x + adjacent.x);
            const float2 dist = fabs(convert_float2(neighbId - adjacent) - 0.5f);
            const float2 weight = 1.0f - half_divide(dist, cellSz);
            interpCellWeights[i] = weight.x * weight.y * 1e6f;
        }
    }

    const int cellCntGlobX = (imGlobSzX - 2 * halfPad.x) / cellSz;
    const int binsPerIter = mul24(mul24(cellCntLoc.y, cellCntGlobX), (int)SENS_BINS);
    int dstIdLoc[2];
    int dstIdGlob[2];
    {
        const int binsCnt = mul24(mul24(cellCntLoc.x, cellCntLoc.y), (int)SENS_BINS);
        const int globShift = mul24((int)get_group_id(0), cellCntLoc.x);
        #pragma unroll
        for (int i = 0; i < 2; ++i)
        {
            dstIdLoc[i] = mad24((int)HOG_WG_SZ_BIG_LIN, i, wiIdLin) % binsCnt;
            const int cellIdLocLin = dstIdLoc[i] / SENS_BINS;
            const int2 cellIdGlob = (int2)(
                cellIdLocLin % cellCntLoc.x + globShift, cellIdLocLin / cellCntLoc.x);
            dstIdGlob[i] = mad24(mad24(cellIdGlob.y, cellCntGlobX, cellIdGlob.x),
                (int)SENS_BINS, dstIdLoc[i] % SENS_BINS);
        }
    }

    for (int iter = 0; iter < iterCnt; ++iter)
    {
        #pragma unroll 2
        for (int i = 0; i < 2; ++i)
        {
            derivsXloc[srcIdLoc[i]] = derivsXglob[srcIdGlob[i]];
            derivsYloc[srcIdLoc[i]] = derivsYglob[srcIdGlob[i]];
            cellDescLoc[dstIdLoc[i]] = 0;
            srcIdGlob[i] += derivsPerIter;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll 4
        for (int i = 0; i < 4; ++i)
        {
            const float2 grad = (float2)(
                derivsXloc[derivIdsLin[i]], derivsYloc[derivIdsLin[i]]);
            const float mag = fast_length(grad) * interpCellWeights[i];
            const float ang = atan2pi(grad.y, grad.x) * 0.5f;
            const float bin = SENS_BINS * (ang + (float)(ang < 0.0f));

            int2 interpBins = (int)bin;
            interpBins.s1++;
            float2 interpBinWeights = bin - (float)interpBins.s0;
            interpBinWeights.s0 = 1.0f - interpBinWeights.s1;
            interpBins = interpBins % SENS_BINS + interpCellId;

            atomic_add(cellDescLoc + interpBins.s0, convert_uint_sat(mag * interpBinWeights.s0));
            atomic_add(cellDescLoc + interpBins.s1, convert_uint_sat(mag * interpBinWeights.s1));
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll 2
        for (int i = 0; i < 2; ++i)
        {
            cellDescGlob[dstIdGlob[i]] = cellDescLoc[dstIdLoc[i]];
            dstIdGlob[i] += binsPerIter;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

inline void loadCellDesc(
    __global const uint* const restrict cellDescGlob,
    float4 sensDesc[5],
    float4 insDesc[3])
{
    #pragma unroll 4
    for (int i = 0; i < 4; ++i)
    {
        sensDesc[i] = convert_float4(vload4(i, cellDescGlob));
    }
    sensDesc[4].s01 = convert_float2(vload2(8, cellDescGlob));
    #pragma unroll 2
    for (int i = 0; i < 2; ++i)
    {
        insDesc[i] = sensDesc[i];
        insDesc[i].s012 += sensDesc[i + 2].s123;
        insDesc[i].s3 += sensDesc[i + 3].s0;
    }
    insDesc[2].s0 = sensDesc[2].s0 + sensDesc[4].s1;
}

__kernel void calcCellNorms(
    __global const uint* restrict cellDescGlob,
    __global float* restrict cellNormsGlob,
    const int iterCnt,
    const int padX)
{
    const int wiIdX = get_local_id(0);
    const int wiIdYglob = get_global_id(1);
    const int cellCntX = mul24((int)HOG_WG_SZ_SMALL, iterCnt);
    const int binCntPerIter = mul24((int)HOG_WG_SZ_SMALL, SENS_BINS);

    cellDescGlob += mul24((int)SENS_BINS, mad24(wiIdYglob, cellCntX, wiIdX));
    cellNormsGlob += mad24(wiIdYglob + 1, cellCntX + padX, wiIdX + 1);

    float4 sensDesc[5];
    float4 insDesc[3];

    for (int iter = 0, shiftDesc = 0, shiftNorms = 0; iter < iterCnt;
         ++iter, shiftDesc += binCntPerIter, shiftNorms += HOG_WG_SZ_SMALL)
    {
        loadCellDesc(cellDescGlob + shiftDesc, sensDesc, insDesc);
        cellNormsGlob[shiftNorms] = insDesc[2].s0 * insDesc[2].s0 +
            dot(insDesc[0], insDesc[0]) + dot(insDesc[1], insDesc[1]);
    }
}

__kernel void sumCellNormsX(
    __global const float* restrict cellNormsGlob,
    __global float* restrict sumsGlob,
    __local float* restrict cellNormsLoc,
    const int iterCnt)
{
    const int wiIdX = get_local_id(0);
    const int copyLoc = wiIdX == 0 ? HOG_WG_SZ_SMALL : 0;

    {
        const int shiftGlob = mad24((int)get_global_id(1),
            mad24((int)HOG_WG_SZ_SMALL, iterCnt, 1), wiIdX);
        cellNormsLoc += mad24((int)get_local_id(1), HOG_WG_SZ_SMALL + 1, wiIdX);
        cellNormsGlob += shiftGlob;
        sumsGlob += shiftGlob;
        *cellNormsLoc = *cellNormsGlob++;
    }

    for (int iter = 0, shiftGlob = 0; iter < iterCnt; ++iter, shiftGlob += HOG_WG_SZ_SMALL)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        const float right = cellNormsGlob[shiftGlob];
        cellNormsLoc[1] = right;
        barrier(CLK_LOCAL_MEM_FENCE);
        sumsGlob[shiftGlob] = *cellNormsLoc + right;
        *cellNormsLoc = cellNormsLoc[copyLoc];
    }
}

__kernel void calcInvBlockNorms(
    __global const float* restrict cellNormsGlob,
    __global float* restrict invBlockNormsGlob,
    __local float* restrict cellNormsLoc,
    const int iterCnt)
{
    const int wiIdY = get_local_id(1);
    const int cellCntGlobX = get_global_size(0) + 1;
    const int copyLoc = wiIdY == 0 ? HOG_WG_SZ_SMALL_LIN : 0;
    const int iterStep = mul24((int)HOG_WG_SZ_SMALL, cellCntGlobX);

    {
        const int shiftGlob = mad24(wiIdY, cellCntGlobX, (int)get_global_id(0));
        cellNormsLoc += mad24(wiIdY, (int)HOG_WG_SZ_SMALL, (int)get_local_id(0));
        cellNormsGlob += shiftGlob;
        invBlockNormsGlob += shiftGlob;
        *cellNormsLoc = *cellNormsGlob;
        cellNormsGlob += cellCntGlobX;
    }

    for (int iter = 0, shiftGlob = 0; iter < iterCnt; ++iter, shiftGlob += iterStep)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        const float bottom = cellNormsGlob[shiftGlob];
        cellNormsLoc[HOG_WG_SZ_SMALL] = bottom;
        barrier(CLK_LOCAL_MEM_FENCE);
        invBlockNormsGlob[shiftGlob] = half_rsqrt(*cellNormsLoc + bottom + 1e-7f);
        *cellNormsLoc = cellNormsLoc[copyLoc];
    }
}

__kernel void applyNormalization(
    __global const uint* restrict cellDescGlob,
    __global const float* restrict invBlockNormsGlob,
    __global float* restrict blockDescGlob,
    __local float* restrict normsLoc,
    const int iterCnt,
    const int padX)
{
    const int2 wiId = (int2)(get_local_id(0), get_local_id(1));
    const int normsLocSzX = HOG_WG_SZ_SMALL + 1;
    const int cellCntGlobX = get_global_size(0);
    const int normsGlobSzX = cellCntGlobX + padX;
    const int normsPerIter = mul24((int)HOG_WG_SZ_SMALL, normsGlobSzX);
    const int cellsPerIter = mul24((int)HOG_WG_SZ_SMALL, cellCntGlobX);
    const int cellBinCntPerIter = mul24(cellsPerIter, (int)SENS_BINS);
    const int blockBinCntPerIter = mul24(cellsPerIter, (int)BINS_PER_BLOCK);

    {
        const int cellIdLin = mad24(wiId.y, cellCntGlobX, (int)get_global_id(0));
        cellDescGlob += mul24(cellIdLin, (int)SENS_BINS);
        blockDescGlob += mul24(cellIdLin, (int)BINS_PER_BLOCK);
    }

    int loadIdLoc[2];
    int loadIdGlob[2];
    {
        const int wiIdLin = mad24(wiId.y, (int)HOG_WG_SZ_SMALL, wiId.x);
        const int normsLocSzLin = mul24(normsLocSzX, HOG_WG_SZ_SMALL + 1);
        loadIdLoc[0] = wiIdLin;
        loadIdLoc[1] = wiIdLin + ((wiIdLin + HOG_WG_SZ_SMALL_LIN < normsLocSzLin) ?
            HOG_WG_SZ_SMALL_LIN : 0);
        #pragma unroll 2
        for (int i = 0; i < 2; ++i)
        {
            const int2 loc = (int2)(loadIdLoc[i] % normsLocSzX, loadIdLoc[i] / normsLocSzX);
            loadIdGlob[i] = mad24(loc.y, normsGlobSzX,
                mad24((int)get_group_id(0), (int)HOG_WG_SZ_SMALL, loc.x));
        }
    }

    int normIds[4];
    {
        normIds[3] = mad24(wiId.y, HOG_WG_SZ_SMALL + 1, wiId.x);
        normIds[2] = normIds[3] + 1;
        normIds[1] = normIds[3] + HOG_WG_SZ_SMALL + 1;
        normIds[0] = normIds[1] + 1;
    }

    float4 sensDesc[5];
    float4 sensDescNorm[5];
    float4 insDesc[3];
    float4 insDescNorm[3];
    float4 tmp;
    float descNorm[4];

    for (int iter = 0, cellDescShift = 0; iter < iterCnt;
         ++iter, cellDescShift += cellBinCntPerIter, blockDescGlob += blockBinCntPerIter)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll 2
        for (int i = 0; i < 2; ++i)
        {
            normsLoc[loadIdLoc[i]] = invBlockNormsGlob[loadIdGlob[i]];
            loadIdGlob[i] += normsPerIter;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        loadCellDesc(cellDescGlob + cellDescShift, sensDesc, insDesc);
        #pragma unroll 4
        for (int i = 0; i < 4; ++i)
        {
            sensDescNorm[i] = 0.0f;
        }
        sensDescNorm[4].s01 = 0.0f;
        #pragma unroll 2
        for (int i = 0; i < 2; ++i)
        {
            insDescNorm[i] = 0.0f;
        }
        insDescNorm[2].s0 = 0.0f;
        #pragma unroll 4
        for (int normId = 0; normId < 4; ++normId)
        {
            const float invNorm = normsLoc[normIds[normId]];
            #pragma unroll 4
            for (int i = 0; i < 4; ++i)
            {
                sensDescNorm[i] += fmin(sensDesc[i] * invNorm, TRUNC);
            }
            sensDescNorm[4].s01 += fmin(sensDesc[4].s01 * invNorm, TRUNC);
            descNorm[normId] = 0.0f;
            #pragma unroll 2
            for (int i = 0; i < 2; ++i)
            {
                tmp = fmin(insDesc[i] * invNorm, TRUNC);
                insDescNorm[i] += tmp;
                descNorm[normId] += tmp.s0 + tmp.s1 + tmp.s2 + tmp.s3;
            }
            tmp.s0 = fmin(insDesc[2].s0 * invNorm, TRUNC);
            insDescNorm[2].s0 += tmp.s0;
            descNorm[normId] += tmp.s0;
        }

        #pragma unroll 4
        for (int i = 0; i < 4; ++i)
        {
            vstore4(sensDescNorm[i] * 0.5f, i, blockDescGlob);
        }
        vstore2(sensDescNorm[4].s01 * 0.5f, 8, blockDescGlob);
        #pragma unroll 2
        for (int i = 0; i < 2; ++i)
        {
            vstore4(insDescNorm[i] * 0.5f, i, blockDescGlob + 18);
        }
        blockDescGlob[26] = insDescNorm[2].s0 * 0.5f;
        #pragma unroll 4
        for (int i = 0; i < 4; ++i)
        {
            blockDescGlob[27 + i] = descNorm[i] * 0.2357f;
        }
    }
}

