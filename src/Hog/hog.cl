inline void calcPartialDerivs2(
    __local const float* const restrict imLoc,
    __local float* const restrict derivX,
    __local float* const restrict derivY,
    const int2 imLocId,
    const int2 derivId,
    const int imLocSzX,
    const int derivSzX)
{
    const int i = mad24(derivId.y, derivSzX, derivId.x);
    const int j = mad24(imLocId.y, imLocSzX, imLocId.x);
    derivX[i] = imLoc[j + 1] - imLoc[j - 1];
    derivY[i] = imLoc[j + imLocSzX] - imLoc[j - imLocSzX];
}

inline int calcOutBinGlob(
    const int wiIdLin,
    const int binsPerCell,
    const int cellSz,
    const int2 cellCntLoc,
    const int imGlobSzX,
    const int halfPadX)
{
    const int cellIdLocLin = wiIdLin / binsPerCell;
    const int binIdLoc = wiIdLin % binsPerCell;
    const int2 cellIdLoc = (int2)(cellIdLocLin % cellCntLoc.x, cellIdLocLin / cellCntLoc.x);
    const int2 cellIdGlob = mad24((int2)(0, get_group_id(1)), cellCntLoc, cellIdLoc);
    const int cellCntGlobX = (imGlobSzX - 2 * halfPadX) / cellSz;
    return mad24(mad24(cellIdGlob.y, cellCntGlobX, cellIdGlob.x), binsPerCell, binIdLoc);
}

__kernel void calcCellDesc(
    __global const float* const restrict imGlob,
    __global uint* const restrict cellDescGlob,
    __local float* const restrict imLoc,
    __local float* const restrict derivX,
    __local float* const restrict derivY,
    __local uint* const restrict cellDescLoc,
    const int iterCnt,
    const int cellSz,
    const int binsPerCell,
    const int2 halfPad)
{
    const int2 wiId = (int2)(get_local_id(0), get_local_id(1));
    const int2 wgSz = (int2)(get_local_size(0), get_local_size(1));
    const int wiIdLin = mad24(wiId.y, wgSz.x, wiId.x);
    const int wgSzLin = mul24(wgSz.x, wgSz.y);
    const int2 imGlobSz = (int2)(mul24(wgSz.x, iterCnt), get_global_size(1)) + 2 * halfPad;
    // TODO reduce bank conflicts
    const int2 imLocSz = wgSz + (int2)(0, cellSz) + 2;
    const int2 derivSz = wgSz + cellSz;
    const int halfCellSz = cellSz / 2;

    // Load data and calculate partial derivatives for the very first image block. This
    // block differs from anothers because the leftmost part of the global image does not
    // exist.
    {
        const int2 imLocId = (int2)(wiIdLin / imLocSz.y, wiIdLin % imLocSz.y);
        const int2 imGlobId = imLocId + (int2)(0, mul24((int)get_group_id(1), wgSz.y)) +
            halfPad - halfCellSz - 1;
        // TODO: when cellSz is 8, not all pixels will be loaded!
        imLoc[mad24(imLocId.y, imLocSz.x, imLocId.x)] =
            imGlob[mad24(imGlobId.y, imGlobSz.x, imGlobId.x)];

        barrier(CLK_LOCAL_MEM_FENCE);
        calcPartialDerivs2(imLoc, derivX, derivY, imLocId + 1, imLocId, imLocSz.x, derivSz.x);

        barrier(CLK_LOCAL_MEM_FENCE);
        const int i = mad24(imLocId.y, imLocSz.x, imLocId.x);
        imLoc[i] = imLoc[i + (imLocId.x < 2 ? cellSz : 0)];
    }

    const int2 imGlobId = halfPad + (int2)(wiId.x + halfCellSz + 1, get_global_id(1));
    const int2 imLocId = wiId + (int2)(0, halfCellSz) + 1;
    const int2 derivId = wiId + (int2)(cellSz, halfCellSz);
    const int load2dir = (wiId.y < halfCellSz + 1) ? -1 : (
        (wiId.y >= wgSz.y - halfCellSz - 1) ? 1 : 0);
    __local float* const locLoad = imLoc + mad24(imLocId.y, imLocSz.x, imLocId.x + 1);
    __local float* const locLoad2 = locLoad + mul24(load2dir, mul24(halfCellSz + 1, imLocSz.x));
    __global const float* const globLoad = imGlob + mad24(imGlobId.y, imGlobSz.x, imGlobId.x);
    __global const float* const globLoad2 = globLoad +
        mul24(load2dir, mul24(halfCellSz + 1, imGlobSz.x));
    const int deriv2dir = wiId.y < halfCellSz ? -1 : ((wiId.y >= wgSz.y - halfCellSz) ? 1 : 0);
    const int2 imLocId2 = imLocId + deriv2dir * (int2)(0, halfCellSz);
    const int2 derivId2 = derivId + deriv2dir * (int2)(0, halfCellSz);

    const int2 neighborId = wiId % cellSz;
    const int2 cellCntLoc = wgSz / cellSz;
    const int2 cellIdLoc = (int2)(wiId.x / cellSz, wiId.y / cellSz);
    __local uint* const dstCellLoc = cellDescLoc +
        mul24(mad24(cellIdLoc.y, cellCntLoc.x, cellIdLoc.x), binsPerCell);
    const float accuracy = 1e6f;

    int derivIdsLin[4];
    float interpCellWeights[4];
    #pragma unroll 4
    for (int i = 0; i < 4; ++i)
    {
        const int2 adjacent = mul24((int2)(i % 2, i / 2), cellSz);
        derivIdsLin[i] = mad24(wiId.y + adjacent.y, derivSz.x, wiId.x + adjacent.x);
        const float2 dist = fabs(convert_float2(cellSz - neighborId - adjacent) - 0.5f);
        const float2 weight = 1.0f - half_divide(dist, cellSz);
        interpCellWeights[i] = weight.x * weight.y * accuracy;
    }

    const int hasChnl2 = wiIdLin + wgSzLin <
        mul24(mul24(cellCntLoc.x, cellCntLoc.y), binsPerCell);
    __local const uint* const dstChnlLoc = cellDescLoc + wiIdLin;
    __local const uint* const dstChnlLoc2 = dstChnlLoc + mul24(hasChnl2, wgSzLin);
    __global uint* const dstChnlGlob = cellDescGlob + calcOutBinGlob(
        wiIdLin, binsPerCell, cellSz, cellCntLoc, imGlobSz.x, halfPad.x);
    __global uint* const dstChnlGlob2 = !hasChnl2 ? dstChnlGlob : (cellDescGlob + calcOutBinGlob(
        wiIdLin + wgSzLin, binsPerCell, cellSz, cellCntLoc, imGlobSz.x, halfPad.x));

    const int2 derivIdCopy = (int2)(wiIdLin % cellSz, wiIdLin / cellSz);
    const int2 imLocIdCopy = (int2)(wiIdLin % 2, wiIdLin / 2);
    __local float* const copyIm = imLoc + mad24(imLocIdCopy.y, imLocSz.x, imLocIdCopy.x);
    __local float* const copyX = derivX + mad24(derivIdCopy.y, derivSz.x, derivIdCopy.x);
    __local float* const copyY = derivY + mad24(derivIdCopy.y, derivSz.x, derivIdCopy.x);

    for (int iter = 0; iter < iterCnt; ++iter)
    {
        // Here we assume that the leftmost parts of the both local image and partial
        // derivatives have been already loaded, so we have to load the rest of data,
        // calculate the rest of derivatives and finally calculate the descriptor for
        // all the cells which are covered by current iteration
        barrier(CLK_LOCAL_MEM_FENCE);
        int tmpId = mul24(iter, wgSz.x);
        *locLoad = globLoad[tmpId];
        *locLoad2 = globLoad2[tmpId];

        barrier(CLK_LOCAL_MEM_FENCE);
        calcPartialDerivs2(imLoc, derivX, derivY, imLocId, derivId, imLocSz.x, derivSz.x);
        calcPartialDerivs2(imLoc, derivX, derivY, imLocId2, derivId2, imLocSz.x, derivSz.x);
        cellDescLoc[wiIdLin] = 0;
        cellDescLoc[wiIdLin + wgSzLin] = 0;

        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll 4
        for (int i = 0; i < 4; ++i)
        {
            const float2 grad = (float2)(derivX[derivIdsLin[i]], derivY[derivIdsLin[i]]);
            const float mag = fast_length(grad) * interpCellWeights[i];
            const float ang = atan2pi(grad.y, grad.x) * 0.5f;
            const float bin = (float)binsPerCell * (ang + (float)(ang < 0.0f));

            int2 interpBins = (int)bin;
            interpBins.s1++;
            float2 interpBinWeights = bin - (float)interpBins.s0;
            interpBinWeights.s0 = 1.0f - interpBinWeights.s1;
            interpBins %= binsPerCell;

            atomic_add(dstCellLoc + interpBins.s0, convert_uint_sat(mag * interpBinWeights.s0));
            atomic_add(dstCellLoc + interpBins.s1, convert_uint_sat(mag * interpBinWeights.s1));
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        tmpId = mul24(mul24(iter, cellCntLoc.x), binsPerCell);
        dstChnlGlob[tmpId] = *dstChnlLoc;
        dstChnlGlob2[tmpId] = *dstChnlLoc2;

        // Prepare to the next iteration: copy parts of local image and its derivatives
        if (derivIdCopy.y < derivSz.y)
        {
            *copyX = copyX[wgSz.x];
            *copyY = copyY[wgSz.x];
        }
        if (imLocIdCopy.y < imLocSz.y)
        {
            *copyIm = copyIm[wgSz.x];
        }
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
    const int sensBinCnt,
    const int padX)
{
    const int wiIdX = get_local_id(0);
    const int wiIdYglob = get_global_id(1);
    const int wgSzX = get_local_size(0);
    const int cellCntX = mul24(wgSzX, iterCnt);
    const int binCntPerIter = mul24(wgSzX, sensBinCnt);

    cellDescGlob += mul24(sensBinCnt, mad24(wiIdYglob, cellCntX, wiIdX));
    cellNormsGlob += mad24(wiIdYglob + 1, cellCntX + padX, wiIdX + 1);

    float4 sensDesc[5];
    float4 insDesc[3];

    for (int iter = 0, shiftDesc = 0, shiftNorms = 0;
         iter < iterCnt;
         ++iter, shiftDesc += binCntPerIter, shiftNorms += wgSzX)
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
    const int wgSzX = get_local_size(0);
    const int copyLoc = wiIdX == 0 ? wgSzX : 0;

    {
        const int shiftGlob = mad24((int)get_global_id(1), mad24(wgSzX, iterCnt, 1), wiIdX);
        cellNormsLoc += mad24((int)get_local_id(1), wgSzX + 1, wiIdX);
        cellNormsGlob += shiftGlob;
        sumsGlob += shiftGlob;
        *cellNormsLoc = *cellNormsGlob++;
    }

    for (int iter = 0, shiftGlob = 0; iter < iterCnt; ++iter, shiftGlob += wgSzX)
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
    const int2 wgSz = (int2)(get_local_size(0), get_local_size(1));
    const int cellCntGlobX = get_global_size(0) + 1;
    const int copyLoc = wiIdY == 0 ? mul24(wgSz.y, wgSz.x) : 0;
    const int iterStep = mul24(wgSz.y, cellCntGlobX);

    {
        const int shiftGlob = mad24(wiIdY, cellCntGlobX, (int)get_global_id(0));
        cellNormsLoc += mad24(wiIdY, wgSz.x, (int)get_local_id(0));
        cellNormsGlob += shiftGlob;
        invBlockNormsGlob += shiftGlob;
        *cellNormsLoc = *cellNormsGlob;
        cellNormsGlob += cellCntGlobX;
    }

    for (int iter = 0, shiftGlob = 0; iter < iterCnt; ++iter, shiftGlob += iterStep)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        const float bottom = cellNormsGlob[shiftGlob];
        cellNormsLoc[wgSz.x] = bottom;
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
    const int padX,
    const int insBinCnt)
{
    const int2 wiId = (int2)(get_local_id(0), get_local_id(1));
    const int2 wgSz = (int2)(get_local_size(0), get_local_size(1));
    const int normsLocSzX = wgSz.x + 1;
    const int cellCntGlobX = mul24(wgSz.x, iterCnt);
    const int sensBinCnt = 2 * insBinCnt;
    const int blockBinCnt = insBinCnt + sensBinCnt + 4;
    const int cellBinCntPerIter = mul24(wgSz.x, sensBinCnt);
    const int blockBinCntPerIter = mul24(wgSz.x, blockBinCnt);
    const float trunc = 0.2f;

    {
        const int cellIdLin = mad24((int)get_global_id(1), cellCntGlobX, wiId.x);
        cellDescGlob += mul24(cellIdLin, sensBinCnt);
        blockDescGlob += mul24(cellIdLin, blockBinCnt);
    }

    int loadIdLoc[2];
    int loadIdGlob[2];
    {
        const int wiIdLin = mad24(wiId.y, wgSz.x, wiId.x);
        const int wgSzLin = mul24(wgSz.x, wgSz.y);
        const int normsLocSzLin = mul24(normsLocSzX, wgSz.y + 1);
        loadIdLoc[0] = wiIdLin;
        loadIdLoc[1] = wiIdLin + ((wiIdLin + wgSzLin < normsLocSzLin) ? wgSzLin : 0);
        #pragma unroll 2
        for (int i = 0; i < 2; ++i)
        {
            const int2 loc = (int2)(loadIdLoc[i] % normsLocSzX, loadIdLoc[i] / normsLocSzX);
            loadIdGlob[i] = mad24(mad24((int)get_group_id(1), wgSz.y, loc.y),
                cellCntGlobX + padX, loc.x);
        }
    }

    int normIds[4];
    {
        normIds[3] = mad24(wiId.y, wgSz.x + 1, wiId.x);
        normIds[2] = normIds[3] + 1;
        normIds[1] = normIds[3] + wgSz.x + 1;
        normIds[0] = normIds[1] + 1;
    }

    float4 sensDesc[5];
    float4 sensDescNorm[5];
    float4 insDesc[3];
    float4 insDescNorm[3];
    float4 tmp;
    float descNorm[4];

    for (int iter = 0, normsShift = 0, cellDescShift = 0;
         iter < iterCnt;
         ++iter, normsShift += wgSz.x, cellDescShift += cellBinCntPerIter,
         blockDescGlob += blockBinCntPerIter)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        normsLoc[loadIdLoc[0]] = invBlockNormsGlob[loadIdGlob[0] + normsShift];
        normsLoc[loadIdLoc[1]] = invBlockNormsGlob[loadIdGlob[1] + normsShift];

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
                sensDescNorm[i] += fmin(sensDesc[i] * invNorm, trunc);
            }
            sensDescNorm[4].s01 += fmin(sensDesc[4].s01 * invNorm, trunc);
            descNorm[normId] = 0.0f;
            #pragma unroll 2
            for (int i = 0; i < 2; ++i)
            {
                tmp = fmin(insDesc[i] * invNorm, trunc);
                insDescNorm[i] += tmp;
                descNorm[normId] += tmp.s0 + tmp.s1 + tmp.s2 + tmp.s3;
            }
            tmp.s0 = fmin(insDesc[2].s0 * invNorm, trunc);
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

