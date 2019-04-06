#define SENS_BINS 18
#define BINS_PER_BLOCK (SENS_BINS * 3 / 2 + 4)

#define HALF_CELL_SZ 2
#define CELL_SZ (HALF_CELL_SZ * 2)
#define TRUNC 0.2f

#define HOG_WG_SZ_BIG 16
#define HOG_WG_SZ_BIG_LIN (HOG_WG_SZ_BIG * HOG_WG_SZ_BIG)

#define CELL_CNT_LOC (HOG_WG_SZ_BIG / CELL_SZ)
#define CELL_CNT_LOC_LIN (CELL_CNT_LOC * CELL_CNT_LOC)
#define BINS_CNT_LOC (CELL_CNT_LOC_LIN * SENS_BINS)

#define HOG_DERIVS_LOC_SZ (HOG_WG_SZ_BIG + CELL_SZ)
#define HOG_DERIVS_LOC_SZ_LIN (HOG_DERIVS_LOC_SZ * HOG_DERIVS_LOC_SZ)

#define HOG_IM_LOC_SZ (HOG_DERIVS_LOC_SZ + 2)
#define HOG_IM_LOC_SZ_LIN (HOG_IM_LOC_SZ * HOG_IM_LOC_SZ)

#define HOG_WG_SZ_SMALL (HOG_WG_SZ_BIG / CELL_SZ)
#define HOG_WG_SZ_SMALL_LIN (HOG_WG_SZ_SMALL * HOG_WG_SZ_SMALL)

#define HOG_WG_SZ_SMALL_PAD (HOG_WG_SZ_SMALL + 1)
#define HOG_WG_SZ_SMALL_PAD_LIN (HOG_WG_SZ_SMALL_PAD * HOG_WG_SZ_SMALL_PAD)
#define HOG_CELL_NORMS_SUM_X_SZ_LIN (HOG_WG_SZ_SMALL * HOG_WG_SZ_SMALL_PAD)

inline void calcDerivsInl(
    __local const float* const restrict im,
    __local float* const restrict derivsX,
    __local float* const restrict derivsY,
    const int imId[2],
    const int derivId[2],
    const int isValid[2])
{
    #pragma unroll 2
    for (int i = 0; i < 2; ++i)
    {
        derivsX[derivId[i]] = !isValid[i] ? 0.0f : (im[imId[i] + 1] - im[imId[i] - 1]);
        derivsY[derivId[i]] = !isValid[i] ? 0.0f :
            (im[imId[i] + HOG_IM_LOC_SZ] - im[imId[i] - HOG_IM_LOC_SZ]);
    }
}

inline void calcCellDescInl(
    __local const float* const restrict derivsX,
    __local const float* const restrict derivsY,
    __local uint* const restrict cellDescLoc,
    __global uint* const restrict cellDescGlob,
    const int derivIdsCell[2],
    const float interpCellWeights[4],
    const int dstIdLoc[2],
    const int interpCellId,
    const int binsPerIter,
    int dstIdGlob[2])
{
    #pragma unroll 2
    for (int i = 0; i < 2; ++i)
    {
        cellDescLoc[dstIdLoc[i]] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    #pragma unroll 4
    for (int i = 0; i < 4; ++i)
    {
        const float2 grad = (float2)(derivsX[derivIdsCell[i]], derivsY[derivIdsCell[i]]);
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
}

__kernel void calcCellDesc(
    __global const float* const restrict imGlob,
    __global uint* const restrict cellDescGlob,
    const int iterCnt)
{
    __local float imLoc[HOG_IM_LOC_SZ_LIN];
    __local float derivsX[HOG_DERIVS_LOC_SZ_LIN];
    __local float derivsY[HOG_DERIVS_LOC_SZ_LIN];
    __local uint cellDescLoc[BINS_CNT_LOC];
    const int2 wiId = (int2)(get_local_id(0), get_local_id(1));
    const int2 imGlobSz = (int2)((int)get_global_size(0), mul24(iterCnt, (int)HOG_WG_SZ_BIG));
    const int wiIdLin = mad24(wiId.y, HOG_WG_SZ_BIG, wiId.x);
    const int imGlobIterStep = mul24((int)HOG_WG_SZ_BIG, imGlobSz.x);
    const int shiftGlobIm = mul24((int)get_group_id(0), (int)HOG_WG_SZ_BIG);

    int srcIdLoc[2];
    int srcIdGlob[2];
    #pragma unroll 2
    for (int i = 0; i < 2; ++i)
    {
        srcIdLoc[i] = mad24(i, HOG_WG_SZ_BIG_LIN, wiIdLin);
        srcIdLoc[i] = srcIdLoc[i] >= HOG_IM_LOC_SZ_LIN ? srcIdLoc[0] : srcIdLoc[i];
        const int2 glob = (int2)(
            srcIdLoc[i] % HOG_IM_LOC_SZ + shiftGlobIm, srcIdLoc[i] / HOG_IM_LOC_SZ) -
            HALF_CELL_SZ - 1;
        srcIdGlob[i] = mad24(glob.y, imGlobSz.x, clamp(glob.x, 0, imGlobSz.x - 1));
    }

    int derivId[2];
    int imLocIdForDeriv[2];
    int isValidDeriv[2];
    #pragma unroll 2
    for (int i = 0; i < 2; ++i)
    {
        derivId[i] = mad24(i, HOG_WG_SZ_BIG_LIN, wiIdLin);
        derivId[i] = derivId[i] >= HOG_DERIVS_LOC_SZ_LIN ? derivId[0] : derivId[i];
        const int2 loc = (int2)(derivId[i] % HOG_DERIVS_LOC_SZ, derivId[i] / HOG_DERIVS_LOC_SZ);
        imLocIdForDeriv[i] = mad24(loc.y + 1, HOG_IM_LOC_SZ, loc.x + 1);
        const int glob = loc.x + shiftGlobIm;
        isValidDeriv[i] = (glob >= HALF_CELL_SZ) & (glob < imGlobSz.x + HALF_CELL_SZ);
    }

    const int2 cellIdLoc = wiId / CELL_SZ;
    const int interpCellId = mul24(mad24(cellIdLoc.y, CELL_CNT_LOC, cellIdLoc.x), (int)SENS_BINS);
    int derivIdsCell[4];
    float interpCellWeights[4];
    {
        const int2 neighbId = CELL_SZ - wiId % CELL_SZ;
        #pragma unroll 4
        for (int i = 0; i < 4; ++i)
        {
            const int2 adjacent = mul24((int2)(i % 2, i / 2), CELL_SZ);
            derivIdsCell[i] = mad24(wiId.y + adjacent.y, HOG_DERIVS_LOC_SZ, wiId.x + adjacent.x);
            const float2 dist = fabs(convert_float2(neighbId - adjacent) - 0.5f);
            const float2 weight = 1.0f - half_divide(dist, CELL_SZ);
            interpCellWeights[i] = weight.x * weight.y * 1e6f;
        }
    }

    const int cellCntGlobX = imGlobSz.x / CELL_SZ;
    const int binsPerIter = mul24((int)(CELL_CNT_LOC * SENS_BINS), cellCntGlobX);
    int dstIdLoc[2];
    int dstIdGlob[2];
    {
        const int shiftGlobCell = mul24((int)get_group_id(0), (int)CELL_CNT_LOC);
        #pragma unroll 2
        for (int i = 0; i < 2; ++i)
        {
            dstIdLoc[i] = mad24(i, HOG_WG_SZ_BIG_LIN, wiIdLin);
            dstIdLoc[i] = dstIdLoc[i] >= BINS_CNT_LOC ? dstIdLoc[0] : dstIdLoc[i];
            const int cellIdLocLin = dstIdLoc[i] / SENS_BINS;
            const int2 cellIdGlob = (int2)(
                cellIdLocLin % CELL_CNT_LOC + shiftGlobCell, cellIdLocLin / CELL_CNT_LOC);
            dstIdGlob[i] = mad24(mad24(cellIdGlob.y, cellCntGlobX, cellIdGlob.x),
                SENS_BINS, dstIdLoc[i] % SENS_BINS);
        }
    }

    #pragma unroll 2
    for (int i = 0; i < 2; ++i)
    {
        imLoc[srcIdLoc[i]] = imGlob[srcIdGlob[i] -
            (srcIdGlob[i] < 0 ? mul24(srcIdGlob[i] / imGlobSz.x - 1, imGlobSz.x) : 0)];
        srcIdGlob[i] += imGlobIterStep;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    {
        int isValidDerivTop[2];
        #pragma unroll 2
        for (int i = 0; i < 2; ++i)
        {
            isValidDerivTop[i] = isValidDeriv[i] &
                (derivId[i] / HOG_DERIVS_LOC_SZ >= HALF_CELL_SZ);
        }
        calcDerivsInl(imLoc, derivsX, derivsY, imLocIdForDeriv, derivId, isValidDerivTop);
    }
    calcCellDescInl(derivsX, derivsY, cellDescLoc, cellDescGlob, derivIdsCell,
        interpCellWeights, dstIdLoc, interpCellId, binsPerIter, dstIdGlob);

    for (int iter = 1; iter + 1 < iterCnt; ++iter)
    {
        #pragma unroll 2
        for (int i = 0; i < 2; ++i)
        {
            imLoc[srcIdLoc[i]] = imGlob[srcIdGlob[i]];
            srcIdGlob[i] += imGlobIterStep;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        calcDerivsInl(imLoc, derivsX, derivsY, imLocIdForDeriv, derivId, isValidDeriv);
        calcCellDescInl(derivsX, derivsY, cellDescLoc, cellDescGlob, derivIdsCell,
            interpCellWeights, dstIdLoc, interpCellId, binsPerIter, dstIdGlob);
    }

    imLoc[srcIdLoc[0]] = imGlob[srcIdGlob[0]];
    imLoc[srcIdLoc[1]] = imGlob[srcIdGlob[1] - (srcIdGlob[1] >= mul24(imGlobSz.x, imGlobSz.y) ?
        mul24(srcIdGlob[1] / imGlobSz.x - imGlobSz.y + 1, imGlobSz.x) : 0)];
    barrier(CLK_LOCAL_MEM_FENCE);
    isValidDeriv[1] &= derivId[1] / HOG_DERIVS_LOC_SZ < HOG_WG_SZ_BIG + HALF_CELL_SZ;
    calcDerivsInl(imLoc, derivsX, derivsY, imLocIdForDeriv, derivId, isValidDeriv);
    calcCellDescInl(derivsX, derivsY, cellDescLoc, cellDescGlob, derivIdsCell,
        interpCellWeights, dstIdLoc, interpCellId, binsPerIter, dstIdGlob);
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
    const int iterCnt)
{
    const int wiIdY = get_local_id(1);
    const int wiIdXglob = get_global_id(0);
    const int cellCntX = get_global_size(0);
    const int normCntX = cellCntX + HOG_WG_SZ_SMALL_PAD;
    const int binCntPerIter = mul24(mul24((int)HOG_WG_SZ_SMALL, cellCntX), SENS_BINS);
    const int normCntPerIter = mul24((int)HOG_WG_SZ_SMALL, normCntX);

    cellDescGlob += mul24((int)SENS_BINS, mad24(wiIdY, cellCntX, wiIdXglob));
    cellNormsGlob += mad24(wiIdY + 1, normCntX, wiIdXglob + 1);

    float4 sensDesc[5];
    float4 insDesc[3];

    for (int iter = 0, shiftDesc = 0, shiftNorms = 0; iter < iterCnt;
         ++iter, shiftDesc += binCntPerIter, shiftNorms += normCntPerIter)
    {
        loadCellDesc(cellDescGlob + shiftDesc, sensDesc, insDesc);
        cellNormsGlob[shiftNorms] = insDesc[2].s0 * insDesc[2].s0 +
            dot(insDesc[0], insDesc[0]) + dot(insDesc[1], insDesc[1]);
    }
}

__kernel void calcInvBlockNorms(
    __global const float* const restrict cellNormsGlob,
    __global float* restrict invBlockNorms,
    const int iterCnt)
{
    __local float cellNormsLoc[HOG_WG_SZ_SMALL_PAD_LIN];
    __local float sumsX[HOG_CELL_NORMS_SUM_X_SZ_LIN];
    const int2 wiId = (int2)(get_local_id(0), get_local_id(1));
    const int cellCntGlobX = get_global_size(0) + 1;
    const int wiIdLin = mad24(wiId.y, HOG_WG_SZ_SMALL, wiId.x);
    const int cellsPerIter = mul24((int)HOG_WG_SZ_SMALL, cellCntGlobX);

    int srcIdLoc[2];
    int srcIdGlob[2];
    {
        const int shiftGlob = mul24((int)get_group_id(0), (int)HOG_WG_SZ_SMALL);
        #pragma unroll 2
        for (int i = 0; i < 2; ++i)
        {
            srcIdLoc[i] = mad24(i, HOG_WG_SZ_SMALL_LIN, wiIdLin);
            srcIdLoc[i] = srcIdLoc[i] >= HOG_WG_SZ_SMALL_PAD_LIN ? srcIdLoc[0] : srcIdLoc[i];
            const int2 loc = (int2)(
                srcIdLoc[i] % HOG_WG_SZ_SMALL_PAD, srcIdLoc[i] / HOG_WG_SZ_SMALL_PAD);
            srcIdGlob[i] = mad24(loc.y, cellCntGlobX, loc.x + shiftGlob);
        }
        invBlockNorms += mad24(wiId.y, cellCntGlobX, wiId.x + shiftGlob);
    }

    int sumIdSrc[2];
    int sumIdDst[2];
    #pragma unroll 2
    for (int i = 0; i < 2; ++i)
    {
        sumIdDst[i] = mad24(i, HOG_WG_SZ_SMALL_LIN, wiIdLin);
        sumIdDst[i] = sumIdDst[i] >= HOG_CELL_NORMS_SUM_X_SZ_LIN ? sumIdDst[0] : sumIdDst[i];
        const int2 loc = (int2)(sumIdDst[i] % HOG_WG_SZ_SMALL, sumIdDst[i] / HOG_WG_SZ_SMALL);
        sumIdSrc[i] = mad24(loc.y, HOG_WG_SZ_SMALL_PAD, loc.x);
    }

    for (int iter = 0; iter < iterCnt; ++iter, invBlockNorms += cellsPerIter)
    {
        #pragma unroll 2
        for (int i = 0; i < 2; ++i)
        {
            cellNormsLoc[srcIdLoc[i]] = cellNormsGlob[srcIdGlob[i]];
            srcIdGlob[i] += cellsPerIter;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll 2
        for (int i = 0; i < 2; ++i)
        {
            sumsX[sumIdDst[i]] = cellNormsLoc[sumIdSrc[i]] + cellNormsLoc[sumIdSrc[i] + 1];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        *invBlockNorms = half_rsqrt(sumsX[wiIdLin] + sumsX[wiIdLin + HOG_WG_SZ_SMALL] + 1e-7f);
    }
}

__kernel void applyNormalization(
    __global const uint* restrict cellDescGlob,
    __global const float* restrict invBlockNormsGlob,
    __global float* restrict blockDescGlob,
    const int iterCnt,
    const int padX)
{
    __local float normsLoc[HOG_WG_SZ_SMALL_PAD_LIN];
    const int2 wiId = (int2)(get_local_id(0), get_local_id(1));
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
        const int normsLocSzLin = HOG_WG_SZ_SMALL_PAD * HOG_WG_SZ_SMALL_PAD;
        loadIdLoc[0] = wiIdLin;
        loadIdLoc[1] = wiIdLin + ((wiIdLin + HOG_WG_SZ_SMALL_LIN < normsLocSzLin) ?
            HOG_WG_SZ_SMALL_LIN : 0);
        #pragma unroll 2
        for (int i = 0; i < 2; ++i)
        {
            const int2 loc = (int2)(
                loadIdLoc[i] % HOG_WG_SZ_SMALL_PAD, loadIdLoc[i] / HOG_WG_SZ_SMALL_PAD);
            loadIdGlob[i] = mad24(loc.y, normsGlobSzX,
                mad24((int)get_group_id(0), (int)HOG_WG_SZ_SMALL, loc.x));
        }
    }

    int normIds[4];
    {
        normIds[3] = mad24(wiId.y, HOG_WG_SZ_SMALL_PAD, wiId.x);
        normIds[2] = normIds[3] + 1;
        normIds[1] = normIds[3] + HOG_WG_SZ_SMALL_PAD;
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

