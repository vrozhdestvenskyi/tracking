inline void calcPartialDerivs2(
    __local const float* restrict imLocal,
    __local float* restrict derivX,
    __local float* restrict derivY,
    const int2 imLocalId,
    const int2 derivId,
    const int imLocalSizeX,
    const int derivSizeX)
{
    const int i = mad24(derivId.y, derivSizeX, derivId.x);
    const int j = mad24(imLocalId.y, imLocalSizeX, imLocalId.x);
    derivX[i] = imLocal[j + 1] - imLocal[j - 1];
    derivY[i] = imLocal[j + imLocalSizeX] - imLocal[j - imLocalSizeX];
}

inline int calcOutputBinGlobal(
    const int wiIdLin,
    const int binsPerCell,
    const int cellSize,
    const int2 cellCountLocal,
    const int imGlobalSizeX,
    const int halfPaddingX)
{
    const int cellIdLocalLin = wiIdLin / binsPerCell;
    const int binIdLocal = wiIdLin % binsPerCell;
    const int2 cellIdLocal = (int2)(
        cellIdLocalLin % cellCountLocal.x,
        cellIdLocalLin / cellCountLocal.x);
    const int2 cellIdGlobal = mad24((int2)(0, get_group_id(1)), cellCountLocal, cellIdLocal);
    const int cellCountGlobalX = (imGlobalSizeX - 2 * halfPaddingX) / cellSize;
    return mad24(mad24(cellIdGlobal.y, cellCountGlobalX, cellIdGlobal.x), binsPerCell, binIdLocal);
}

__kernel void calculateCellDescriptor(
    __global const float* restrict imGlobal,
    __global unsigned int* restrict cellDescriptorGlobal,
    __local float* restrict imLocal,
    __local float* restrict derivX,
    __local float* restrict derivY,
    __local unsigned int* restrict cellDescriptorLocal,
    const int iterationsCount,
    const int cellSize,
    const int binsPerCell,
    const int2 halfPadding)
{
    const int2 wiId = (int2)(get_local_id(0), get_local_id(1));
    const int2 wgSize = (int2)(get_local_size(0), get_local_size(1));
    const int wiIdLin = mad24(wiId.y, wgSize.x, wiId.x);
    const int wgSizeLin = mul24(wgSize.x, wgSize.y);
    const int2 imGlobalSize = (int2)(mul24(wgSize.x, iterationsCount), get_global_size(1)) +
        2 * halfPadding;
    // TODO reduce bank conflicts
    const int2 imLocalSize = wgSize + (int2)(0, cellSize) + 2;
    const int2 derivSize = wgSize + cellSize;
    const int halfCellSize = cellSize / 2;

    // Load data and calculate partial derivatives for the very first image block. This
    // block differs from anothers because the leftmost part of the global image does not
    // exist.
    {
        const int2 imLocalId = (int2)(wiIdLin / imLocalSize.y, wiIdLin % imLocalSize.y);
        const int2 imGlobalId = imLocalId + (int2)(0, mul24((int)get_group_id(1), wgSize.y)) +
            halfPadding - halfCellSize - 1;
        imLocal[mad24(imLocalId.y, imLocalSize.x, imLocalId.x)] =
            imGlobal[mad24(imGlobalId.y, imGlobalSize.x, imGlobalId.x)];

        barrier(CLK_LOCAL_MEM_FENCE);
        if (imLocalId.x > 0 && imLocalId.y > 0 &&
            imLocalId.x + 1 < cellSize + 2 && imLocalId.y + 1 < imLocalSize.y)
        {
            calcPartialDerivs2(imLocal, derivX, derivY, imLocalId, imLocalId - 1,
                imLocalSize.x, derivSize.x);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        if (imLocalId.x < 2)
        {
            const int i = mad24(imLocalId.y, imLocalSize.x, imLocalId.x);
            imLocal[i] = imLocal[i + cellSize];
        }
    }

    const int2 imGlobalId = halfPadding + (int2)(wiId.x + halfCellSize + 1, get_global_id(1));
    const int2 imLocalId = wiId + (int2)(0, halfCellSize) + 1;
    const int2 derivId = wiId + (int2)(cellSize, halfCellSize);
    const int load2dir = wiId.y < halfCellSize + 1 ? -1 :
        (wiId.y >= wgSize.y - halfCellSize - 1 ? 1 : 0);
    __local float* const locLoad = imLocal + mad24(imLocalId.y, imLocalSize.x, imLocalId.x + 1);
    __local float* const locLoad2 = locLoad +
        mul24(load2dir, mul24(halfCellSize + 1, imLocalSize.x));
    __global const float* const globLoad = imGlobal +
        mad24(imGlobalId.y, imGlobalSize.x, imGlobalId.x);
    __global const float* const globLoad2 = globLoad +
        mul24(load2dir, mul24(halfCellSize + 1, imGlobalSize.x));
    const int deriv2dir = wiId.y < halfCellSize ? -1 : (wiId.y >= wgSize.y - halfCellSize ? 1 : 0);
    const int2 imLocalId2 = imLocalId + deriv2dir * (int2)(0, halfCellSize);
    const int2 derivId2 = derivId + deriv2dir * (int2)(0, halfCellSize);

    const int2 neighborId = wiId % cellSize;
    const int2 cellCountLocal = wgSize / cellSize;
    const int2 cellIdLocal = (int2)(wiId.x / cellSize, wiId.y / cellSize);
    __local unsigned int* const dstCellLocal = cellDescriptorLocal +
        mul24(mad24(cellIdLocal.y, cellCountLocal.x, cellIdLocal.x), binsPerCell);
    const float accuracy = 1e6f;

    int derivIdsLin[4];
    float interpCellWeights[4];
    for (int i = 0; i < 4; ++i)
    {
        const int2 adjacent = mul24((int2)(i % 2, i / 2), cellSize);
        derivIdsLin[i] = mad24(wiId.y + adjacent.y, derivSize.x, wiId.x + adjacent.x);
        const float2 dist = fabs(convert_float2(cellSize - neighborId - adjacent) - 0.5f);
        const float2 weight = 1.0f - half_divide(dist, cellSize);
        interpCellWeights[i] = weight.x * weight.y * accuracy;
    }

    const int hasChannel2 = wiIdLin + wgSizeLin <
        mul24(mul24(cellCountLocal.x, cellCountLocal.y), binsPerCell);
    __local const unsigned int* const dstChannelLocal = cellDescriptorLocal + wiIdLin;
    __local const unsigned int* const dstChannelLocal2 = dstChannelLocal +
        mul24(hasChannel2, wgSizeLin);
    __global unsigned int* const dstChannelGlobal = cellDescriptorGlobal + calcOutputBinGlobal(
        wiIdLin, binsPerCell, cellSize, cellCountLocal, imGlobalSize.x, halfPadding.x);
    __global unsigned int* const dstChannelGlobal2 = !hasChannel2 ? dstChannelGlobal :
        (cellDescriptorGlobal + calcOutputBinGlobal(
            wiIdLin + wgSizeLin, binsPerCell, cellSize, cellCountLocal,
            imGlobalSize.x, halfPadding.x));

    const int2 derivIdCopy = (int2)(wiIdLin % cellSize, wiIdLin / cellSize);
    const int2 imLocalIdCopy = (int2)(wiIdLin % 2, wiIdLin / 2);
    __local float* const copyIm = imLocal + mad24(imLocalIdCopy.y, imLocalSize.x, imLocalIdCopy.x);
    __local float* const copyX = derivX + mad24(derivIdCopy.y, derivSize.x, derivIdCopy.x);
    __local float* const copyY = derivY + mad24(derivIdCopy.y, derivSize.x, derivIdCopy.x);

    for (int iteration = 0; iteration < iterationsCount; ++iteration)
    {
        // Here we assume that the leftmost parts of the both local image and partial
        // derivatives have been already loaded, so we have to load the rest of data,
        // calculate the rest of derivatives and finally calculate the descriptor for
        // all the cells which are covered by current iteration
        barrier(CLK_LOCAL_MEM_FENCE);
        int tmpId = mul24(iteration, wgSize.x);
        *locLoad = globLoad[tmpId];
        *locLoad2 = globLoad2[tmpId];

        barrier(CLK_LOCAL_MEM_FENCE);
        calcPartialDerivs2(imLocal, derivX, derivY, imLocalId, derivId, imLocalSize.x, derivSize.x);
        calcPartialDerivs2(imLocal, derivX, derivY, imLocalId2, derivId2, imLocalSize.x, derivSize.x);
        cellDescriptorLocal[wiIdLin] = 0;
        cellDescriptorLocal[wiIdLin + wgSizeLin] = 0;

        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
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

            atomic_add(dstCellLocal + interpBins.s0, convert_uint_sat(mag * interpBinWeights.s0));
            atomic_add(dstCellLocal + interpBins.s1, convert_uint_sat(mag * interpBinWeights.s1));
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        tmpId = mul24(mul24(iteration, cellCountLocal.x), binsPerCell);
        dstChannelGlobal[tmpId] = *dstChannelLocal;
        dstChannelGlobal2[tmpId] = *dstChannelLocal2;

        // Prepare to the next iteration: copy parts of local image and its derivatives
        if (derivIdCopy.y < derivSize.y)
        {
            *copyX = copyX[wgSize.x];
            *copyY = copyY[wgSize.x];
        }
        if (imLocalIdCopy.y < imLocalSize.y)
        {
            *copyIm = copyIm[wgSize.x];
        }
    }
}

