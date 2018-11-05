inline void calcPartialDerivs2(
    __local const float* restrict imLocal,
    __local float* restrict derivX,
    __local float* restrict derivY,
    const int2 imLocalId,
    const int2 derivId,
    const int imLocalSizeX,
    const int derivSizeX)
{
    const int i = derivId.x + derivId.y * derivSizeX;
    const int j = imLocalId.x + imLocalId.y * imLocalSizeX;
    derivX[i] = imLocal[j + 1] - imLocal[j - 1];
    derivY[i] = imLocal[j + imLocalSizeX] - imLocal[j - imLocalSizeX];
}

__kernel void calculateCellDescriptor(
    __global const float* restrict imGlobal,
    __global float* restrict cellDescriptorGlobal,
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
    const int wiIdLin = wiId.x + wiId.y * wgSize.x;
    const int2 imGlobalSize = (int2)(wgSize.x * iterationsCount, get_global_size(1)) + 2 * halfPadding;
    // TODO reduce bank conflicts
    const int2 imLocalSize = wgSize + (int2)(0, cellSize) + 2;
    const int2 derivSize = wgSize + cellSize;
    const int halfCellSize = cellSize / 2;

    // Load data and calculate partial derivatives for the very first image block. This
    // block differs from anothers because the leftmost part of the global image does not
    // exist.
    {
        const int initSizeX = cellSize + 2;
        const int2 imLocalId = (int2)(wiIdLin % initSizeX, wiIdLin / initSizeX);
        const int2 imGlobalId = imLocalId + (int2)(0, get_group_id(1) * wgSize.y) +
            halfPadding - halfCellSize - 1;
        // TODO when cellSize is 8, not all pixels will be loaded
        if (imLocalId.y < imLocalSize.y)
        {
            imLocal[imLocalId.x + imLocalId.y * imLocalSize.x] =
                imGlobal[imGlobalId.x + imGlobalId.y * imGlobalSize.x];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        if (imLocalId.x > 0 && imLocalId.y > 0 &&
            imLocalId.x + 1 < initSizeX && imLocalId.y + 1 < imLocalSize.y)
        {
            calcPartialDerivs2(imLocal, derivX, derivY, imLocalId, imLocalId - 1,
                imLocalSize.x, derivSize.x);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        if (imLocalId.x < 2 && imLocalId.y < imLocalSize.y)
        {
            const int i = imLocalId.x + imLocalId.y * imLocalSize.x;
            imLocal[i] = imLocal[i + cellSize];
        }
    }

    const int2 imGlobalId = halfPadding + (int2)(wiId.x + halfCellSize + 1, get_global_id(1));
    const int2 imLocalId = wiId + (int2)(0, halfCellSize) + 1;
    const int2 derivId = wiId + (int2)(cellSize, halfCellSize);
    __local float* locMid = imLocal + imLocalId.x + 1 + imLocalId.y * imLocalSize.x;
    __local float* locTop = locMid - (halfCellSize + 1) * imLocalSize.x;
    __local float* locBot = locMid + (halfCellSize + 1) * imLocalSize.x;
    __global const float* globMid = imGlobal + imGlobalId.x + imGlobalId.y * imGlobalSize.x;
    __global const float* globTop = globMid - (halfCellSize + 1) * imGlobalSize.x;
    __global const float* globBot = globMid + (halfCellSize + 1) * imGlobalSize.x;

    const float accuracy = 1e6f;

    const int2 derivIdCopy = (int2)(wiIdLin % cellSize, wiIdLin / cellSize);
    const int2 imLocalIdCopy = (int2)(wiIdLin % 2, wiIdLin / 2);
    __local float* copyIm = imLocal + imLocalIdCopy.x + imLocalIdCopy.y * imLocalSize.x;
    __local float* copyX = derivX + derivIdCopy.x + derivIdCopy.y * derivSize.x;
    __local float* copyY = derivY + derivIdCopy.x + derivIdCopy.y * derivSize.x;

    for (int iteration = 0; iteration < iterationsCount; ++iteration)
    {
        // Here we assume that the leftmost parts of the both local image and partial
        // derivatives have been already loaded, so we have to load the rest of data,
        // calculate the rest of derivatives and finally calculate the descriptor for
        // all the cells which are covered by current iteration
        {
            barrier(CLK_LOCAL_MEM_FENCE);
            *locMid = *globMid;
            if (wiId.y < halfCellSize + 1)
            {
                *locTop = *globTop;
            }
            if (wiId.y >= wgSize.y - halfCellSize - 1)
            {
                *locBot = *globBot;
            }
            globMid += wgSize.y;
            globTop += wgSize.y;
            globBot += wgSize.y;

            barrier(CLK_LOCAL_MEM_FENCE);
            calcPartialDerivs2(imLocal, derivX, derivY, imLocalId, derivId, imLocalSize.x, derivSize.x);
            if (wiId.y < halfCellSize)
            {
                calcPartialDerivs2(imLocal, derivX, derivY,
                    imLocalId - (int2)(0, halfCellSize),
                    derivId - (int2)(0, halfCellSize),
                    imLocalSize.x, derivSize.x);
            }
            if (wiId.y >= wgSize.y - halfCellSize)
            {
                calcPartialDerivs2(imLocal, derivX, derivY,
                    imLocalId + (int2)(0, halfCellSize),
                    derivId + (int2)(0, halfCellSize),
                    imLocalSize.x, derivSize.x);
            }

            const int2 cellsCountLocal = wgSize / cellSize;
            const int binsCountLocal = cellsCountLocal.x * cellsCountLocal.y * binsPerCell;
            const int wgSizeLin = wgSize.x * wgSize.y;
            cellDescriptorLocal[wiIdLin] = 0;
            if (wiIdLin + wgSizeLin < binsCountLocal)
            {
                cellDescriptorLocal[wiIdLin + wgSizeLin] = 0;
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            {
                const int2 cellId = wiId / cellSize;
                const int2 neighborId = wiId % cellSize;
                const int2 cellCountLocal = wgSize / cellSize;

                for (int dy = 0; dy <= cellSize; dy += cellSize)
                {
                    for (int dx = 0; dx <= cellSize; dx += cellSize)
                    {
                        const int derivIdLin = wiId.x + dx + (wiId.y + dy) * derivSize.x;
                        const float2 gradient = (float2)(derivX[derivIdLin], derivY[derivIdLin]);
                        const float angle = atan2pi(gradient.y, gradient.x) * 0.5f;
                        const float bin = (float)binsPerCell * (angle + (float)(angle < 0.0f));

                        int2 interpBins = (int)bin;
                        interpBins.s1++;
                        float2 interpBinWeights = bin - (float)interpBins.s0;
                        interpBinWeights.s0 = 1.0f - interpBinWeights.s1;
                        interpBins %= binsPerCell;
                        interpBins += (cellId.x + cellId.y * cellCountLocal.x) * binsPerCell;

                        const float2 distance = fabs(convert_float2(cellSize - neighborId - (int2)(dx, dy)) - 0.5f);
                        const float2 interpCellWeights = 1.0f - distance / cellSize;
                        // TODO fast_length
                        const float magnitude = length(gradient) * interpCellWeights.x * interpCellWeights.y;

                        atomic_add(cellDescriptorLocal + interpBins.s0,
                            convert_uint_sat(magnitude * interpBinWeights.s0 * accuracy));
                        atomic_add(cellDescriptorLocal + interpBins.s1,
                            convert_uint_sat(magnitude * interpBinWeights.s1 * accuracy));
                    }
                }

                /*const int8 derivIds = wiId + (int8)(0, 0, cellSize, 0, 0, cellSize, cellSize, cellSize);
                const int4 derivIdsLin = derivIds.even + derivIds.odd * derivSize.x;

                const float8 grads = (float8)(
                    derivX[derivIdLin.s0], derivY[derivIdLin.s0],
                    derivX[derivIdLin.s1], derivY[derivIdLin.s1],
                    derivX[derivIdLin.s2], derivY[derivIdLin.s2],
                    derivX[derivIdLin.s3], derivY[derivIdLin.s3]);
                const float4 angs = atan2pi(grads.odd, grads.even) * 0.5;
                const float4 bins = (float4)(binsPerCell) * (angs + (float4)(angs < 0.0f));

                int8 interpBins = (int8)(bins, bins);
                interpBins.odd++;
                float8 interpBinWeights = bins - (float)interpBins.s0;
                interpBinWeights.s0 = 1.0f - interpBinWeights.s1;
                interpBins %= binsPerCell;
                interpBins += (cellId.x + cellId.y * cellCountLocal.x) * binsPerCell;*/
            }
            // TODO should we calculate insensitive bins in this kernel too?

            //
            barrier(CLK_LOCAL_MEM_FENCE);
            {
                const int cellIdLocalLin = wiIdLin / binsPerCell;
                const int binIdLocal = wiIdLin % binsPerCell;
                const int2 cellIdLocal = (int2)(
                    cellIdLocalLin % cellsCountLocal.x,
                    cellIdLocalLin / cellsCountLocal.x);
                const int2 cellIdGlobal = (int2)(iteration, get_group_id(1)) * cellsCountLocal + cellIdLocal;
                const int2 cellCountGlobal = (imGlobalSize - 2 * halfPadding) / cellSize;

                const int i = (cellIdGlobal.x + cellIdGlobal.y * cellCountGlobal.x) * binsPerCell;
                cellDescriptorGlobal[i + binIdLocal] = cellDescriptorLocal[wiIdLin] / accuracy;
            }
            {
                const int cellIdLocalLin = (wiIdLin + wgSizeLin) / binsPerCell;
                const int binIdLocal = (wiIdLin + wgSizeLin) % binsPerCell;
                const int2 cellIdLocal = (int2)(
                    cellIdLocalLin % cellsCountLocal.x,
                    cellIdLocalLin / cellsCountLocal.x);
                const int2 cellIdGlobal = (int2)(iteration, get_group_id(1)) * cellsCountLocal + cellIdLocal;
                const int2 cellCountGlobal = (imGlobalSize - 2 * halfPadding) / cellSize;

                if (wiIdLin + wgSizeLin < binsCountLocal)
                {
                    const int i = (cellIdGlobal.x + cellIdGlobal.y * cellCountGlobal.x) * binsPerCell;
                    cellDescriptorGlobal[i + binIdLocal] = cellDescriptorLocal[wiIdLin + wgSizeLin] /
                        accuracy;
                }
            }
        }

        // Prepare to the next iteration: copy parts of local image and its derivatives
        barrier(CLK_LOCAL_MEM_FENCE);
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

