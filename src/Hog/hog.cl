__kernel void calcDerivsX(
    __global const float* restrict imGlob,
    __global float* restrict derivs,
    __local float* restrict imLoc, // column-major
    const int iterCnt,
    const int halfPad)
{
    const int2 wiId = (int2)(get_local_id(1), get_local_id(0));
    const int2 wgSz = (int2)(get_local_size(1), get_local_size(0));
    const int imGlobSzX = mul24(wgSz.x, iterCnt);
    const int copyId = ((wiId.x < 2 ? wgSz.x : 0) - 1) * wgSz.y;

    {
        const int wiIdYglob = get_global_id(0);
        imGlob += mad24(wiIdYglob, imGlobSzX, wiId.x);
        derivs += mad24(wiIdYglob + halfPad, imGlobSzX + 2 * halfPad, wiId.x + halfPad);
        imLoc += mad24(wiId.x + 1, wgSz.y, wiId.y);
        imLoc[-wgSz.y] = imGlob[wiId.x > 0 ? -1 : 0];
    }

    int shiftGlob = 0;
    for (int iter = 0; iter + 1 < iterCnt; ++iter, shiftGlob += wgSz.x)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        imLoc[wgSz.y] = imGlob[shiftGlob + 1];
        barrier(CLK_LOCAL_MEM_FENCE);
        derivs[shiftGlob] = imLoc[wgSz.y] - imLoc[-wgSz.y];
        barrier(CLK_LOCAL_MEM_FENCE); // TODO remove this
        imLoc[-wgSz.y] = imLoc[copyId];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    imLoc[wgSz.y] = imGlob[shiftGlob + (wiId.x + 1 < wgSz.x)];
    barrier(CLK_LOCAL_MEM_FENCE);
    derivs[shiftGlob] = imLoc[wgSz.y] - imLoc[-wgSz.y];
}

__kernel void calcDerivsY(
    __global const float* restrict imGlob,
    __global float* restrict derivs,
    __local float* restrict imLoc,
    const int iterCnt,
    const int halfPad)
{
    const int2 wiId = (int2)(get_local_id(0), get_local_id(1));
    const int2 wgSz = (int2)(get_local_size(0), get_local_size(1));
    const int imGlobSzX = get_global_size(0);
    const int imGlobIterStep = mul24(wgSz.y, imGlobSzX);
    const int derivsIterStep = mul24(wgSz.y, imGlobSzX + 2 * halfPad);
    const int copyId = ((wiId.y < 2 ? wgSz.x : 0) - 1) * wgSz.x;

    {
        const int wiIdXglob = get_global_id(0);
        imGlob += mad24(wiId.y, imGlobSzX, wiIdXglob);
        derivs += mad24(wiId.y + halfPad, imGlobSzX + 2 * halfPad, wiIdXglob + halfPad);
        imLoc += mad24(wiId.y + 1, wgSz.x, wiId.x);
        imLoc[-wgSz.x] = imGlob[wiId.y > 0 ? -imGlobSzX : 0];
    }

    int imGlobShift, derivsShift = 0;
    for (int iter = 0; iter + 1 < iterCnt;
         ++iter, imGlobShift += imGlobIterStep, derivsShift += derivsIterStep)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        imLoc[wgSz.x] = imGlob[imGlobShift + imGlobSzX];
        barrier(CLK_LOCAL_MEM_FENCE);
        derivs[derivsShift] = imLoc[wgSz.x] - imLoc[-wgSz.x];
        barrier(CLK_LOCAL_MEM_FENCE); // TODO remove this
        imLoc[-wgSz.x] = imLoc[copyId];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    imLoc[wgSz.x] = imGlob[imGlobShift + ((wiId.y + 1) < wgSz.y ? imGlobSzX : 0)];
    barrier(CLK_LOCAL_MEM_FENCE);
    derivs[derivsShift] = imLoc[wgSz.x] - imLoc[-wgSz.x];
}

inline void calcDerivsInl(
    __local const float* const restrict im,
    __global float* const restrict derivX,
    __global float* const restrict derivY,
    const int derivId,
    const int imSzX)
{
    derivX[derivId] = im[1] - im[-1];
    derivY[derivId] = im[imSzX] - im[-imSzX];
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
    const int2 wgSz = (int2)(get_local_size(0), get_local_size(1));
    const int2 imLocSz = wgSz + 2;
    const int2 imGlobSz = (int2)(get_global_size(0), mul24(wgSz.y, iterCnt));
    const int derivsSzX = imGlobSz.x + 2 * halfPad;
    const int derivIdLoc = mad24(wiId.y + 1, imLocSz.x, wiId.x + 1);
    const int imGlobIterStep = mul24(wgSz.y, imGlobSz.x);
    const int derivsIterStep = mul24(wgSz.y, derivsSzX);

    int derivsShift = mad24(wiId.y + halfPad, derivsSzX, (int)get_global_id(0) + halfPad);
    int srcIdLoc[2];
    int srcIdGlob[2];
    {
        const int wiIdLin = mad24(wiId.y, wgSz.x, wiId.x);
        const int wgSzLin = mul24(wgSz.x, wgSz.y);
        const int imLocSzLin = mul24(imLocSz.x, imLocSz.y);
        const int wgShiftX = mul24((int)get_group_id(0), wgSz.x);
        #pragma unroll 2
        for (int i = 0; i < 2; ++i)
        {
            srcIdLoc[i] = (wiIdLin + mul24(i, wgSzLin)) % imLocSzLin;
            srcIdGlob[i] = mad24(srcIdLoc[i] / imLocSz.x - 1, imGlobSz.x,
                clamp(srcIdLoc[i] % imLocSz.x + wgShiftX - 1, 0, imGlobSz.x - 1));
        }
    }

    #pragma unroll 2
    for (int i = 0; i < 2; ++i)
    {
        imLoc[srcIdLoc[i]] = imGlob[srcIdGlob[i] + (srcIdGlob[i] >= 0 ? 0 : imGlobSz.x)];
        srcIdGlob[i] += imGlobIterStep;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    calcDerivsInl(imLoc + derivIdLoc, derivsX, derivsY, derivsShift, imLocSz.x);
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
        calcDerivsInl(imLoc + derivIdLoc, derivsX, derivsY, derivsShift, imLocSz.x);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    imLoc[srcIdLoc[0]] = imGlob[srcIdGlob[0]];
    imLoc[srcIdLoc[1]] = imGlob[srcIdGlob[1] -
        (srcIdGlob[1] < mul24(imGlobSz.x, imGlobSz.y) ? 0 : imGlobSz.x)];
    barrier(CLK_LOCAL_MEM_FENCE);
    calcDerivsInl(imLoc + derivIdLoc, derivsX, derivsY, derivsShift, imLocSz.x);
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
    __global const float* restrict derivsXglob,
    __global const float* restrict derivsYglob,
    __global uint* const restrict cellDescGlob,
    __local float* const restrict derivsXloc,
    __local float* const restrict derivsYloc,
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
    const int2 imGlobSz = (int2)(mul24(iterCnt, wgSz.x), (int)get_global_size(1)) + 2 * halfPad;
    const int2 derivsSzGlob = imGlobSz + cellSz;
    const int2 derivsSzLoc = wgSz + cellSz;
    const int halfCellSz = cellSz / 2;

    int srcIdLoc[2];
    int srcIdGlob[2];
    {
        const int derivsSzLocLin = mul24(derivsSzLoc.x, derivsSzLoc.y);
        const int wgShiftY = mul24((int)get_group_id(1), wgSz.y);
        #pragma unroll 2
        for (int i = 0; i < 2; ++i)
        {
            srcIdLoc[i] = (wiIdLin + mul24(i, wgSzLin)) % derivsSzLocLin;
            srcIdGlob[i] = mad24(srcIdLoc[i] / derivsSzLoc.x + wgShiftY + halfPad.y,
                derivsSzGlob.x, srcIdLoc[i] % derivsSzLoc.x + halfPad.x);
        }
    }

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
        derivIdsLin[i] = mad24(wiId.y + adjacent.y, derivsSzLoc.x, wiId.x + adjacent.x);
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

    for (int iter = 0; iter < iterCnt; ++iter)
    {
        #pragma unroll 2
        for (int i = 0; i < 2; ++i)
        {
            derivsXloc[srcIdLoc[i]] = derivsXglob[srcIdGlob[i]];
            derivsYloc[srcIdLoc[i]] = derivsYglob[srcIdGlob[i]];
            srcIdGlob[i] += wgSz.x;
        }
        cellDescLoc[wiIdLin] = 0;
        cellDescLoc[wiIdLin + wgSzLin] = 0;
        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll 4
        for (int i = 0; i < 4; ++i)
        {
            const float2 grad = (float2)(
                derivsXloc[derivIdsLin[i]], derivsYloc[derivIdsLin[i]]);
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

        const int tmpId = mul24(mul24(iter, cellCntLoc.x), binsPerCell);
        dstChnlGlob[tmpId] = *dstChnlLoc;
        dstChnlGlob2[tmpId] = *dstChnlLoc2;
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

