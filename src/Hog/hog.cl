inline void atomicAddLF(volatile __local float *addr, float val)
{
    union {
        unsigned int u32;
        float f32;
    } next, expected, current;
    current.f32 = *addr;
    do {
        expected.f32 = current.f32;
        next.f32 = expected.f32 + val;
        current.u32 = atomic_cmpxchg(
            (volatile __local unsigned int *)addr, expected.u32, next.u32);
    } while (current.u32 != expected.u32);
}

inline void calcPartialDerivs2(
    __local const float* restrict imLocal,
    __local float* restrict derivX,
    __local float* restrict derivY,
    const int2 derivId,
    const int2 imLocalId,
    const int derivSizeX,
    const int imLocalSizeX)
{
    const int i = derivId.x + derivId.y * derivSizeX;
    const int j = imLocalId.x + imLocalId.y * imLocalSizeX;
    derivX[i] = imLocal[j + 1] - imLocal[j - 1];
    derivY[i] = imLocal[j + imLocalSizeX] - imLocal[j - imLocalSizeX];
}

inline void interpHog(
    __local const float* restrict derivX,
    __local const float* restrict derivY,
    __local float* restrict cellDescriptorLocal,
    const int cellSize,
    const int binsPerCell)
{
    const int2 wiId = (int2)(get_local_id(0), get_local_id(1));
    const int2 cellId = wiId / cellSize;
    const int2 neighborId = wiId % cellSize;

    const int2 wgSize = (int2)(get_local_size(0), get_local_size(1));
    const int2 derivSize = wgSize + cellSize;
    const int2 cellCountLocal = wgSize / cellSize;

    for (int dy = 0; dy <= cellSize; dy += cellSize)
    {
        for (int dx = 0; dx <= cellSize; dx += cellSize)
        {
            const int derivIdLin = wiId.x + dx + (wiId.y + dy) * derivSize.x;
            const float2 gradient = (float2)(derivX[derivIdLin], derivY[derivIdLin]);
            const float magnitude = length(gradient); // TODO fast_length
            const float angle = atan2pi(gradient.y, gradient.x) * 0.5f;
            const float bin = (float)binsPerCell * (angle + (float)(angle < 0.0f));

            int2 interpBins = (int)bin;
            interpBins.s1++;
            float2 interpBinWeights = bin - (float)interpBins.s0;
            interpBinWeights.s0 = 1.0f - interpBinWeights.s0;
            interpBins %= binsPerCell;

            const float2 distance = fabs(convert_float2(cellSize - neighborId) - 0.5f);
            const float2 interpCellWeights = 1.0f - distance / cellSize;

            interpBins += (cellId.x + cellId.y * cellCountLocal.x) * binsPerCell;

            atomicAddLF(cellDescriptorLocal + interpBins.s0,
                magnitude * interpCellWeights.s0 * interpBinWeights.s0);
            atomicAddLF(cellDescriptorLocal + interpBins.s1,
                magnitude * interpCellWeights.s1 * interpBinWeights.s1);
        }
    }
}

void performHogInit(
    __global const float* imGlobal,
    __local float* restrict imLocal,
    __local float* restrict derivX,
    __local float* restrict derivY,
    const int cellSize,
    const int iterationsCount)
{
    const int2 wiId = (int2)(get_local_id(0), get_local_id(1));
    const int2 wgSize = (int2)(get_local_size(0), get_local_size(1));
    const int halfCellSize = cellSize / 2;
    const int2 imGlobalSize = (int2)(wgSize.x * iterationsCount, get_global_size(1));

    // TODO reduce bank conflicts
    const int2 derivSize = wgSize + cellSize;
    const int2 imLocalSize = wgSize + (int2)(0, cellSize) + 2;

    const int wiIdLinear = wiId.x + wiId.y * wgSize.x;
    const int2 derivId = (int2)(wiIdLinear % halfCellSize, wiIdLinear / halfCellSize);
    if (derivId.y < derivSize.y)
    {
        const int i = derivId.x + derivId.y * derivSize.x;
        derivX[i] = 0.0f;
        derivY[i] = 0.0f;
    }

    const int initSizeX = halfCellSize + 2;
    const int2 imLocalId = (int2)(wiIdLinear % initSizeX, wiIdLinear / initSizeX);
    if (imLocalId.y < imLocalSize.y)
    {
        const int2 imGlobalId = clamp(
            imLocalId + (int2)(0, get_group_id(1) * wgSize.y - halfCellSize) - 1,
            (int2)(0, 0),
            imGlobalSize - 1);
        imLocal[imLocalId.x + imLocalId.y * imLocalSize.x] =
            imGlobal[imGlobalId.x + imGlobalId.y * imGlobalSize.x];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (imLocalId.x > 0 && imLocalId.y > 0 &&
        imLocalId.x + 1 < imLocalSize.x && imLocalId.y + 1 < imLocalSize.y)
    {
        calcPartialDerivs2(imLocal, derivX, derivY,
            derivId + halfCellSize, imLocalId, derivSize.x, imLocalSize.x);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (imLocalId.x < 2 && imLocalId.y < imLocalSize.y)
    {
        const int i = imLocalId.x + imLocalId.y * imLocalSize.x;
        imLocal[i + halfCellSize] = imLocal[i];
    }
}

void performHogIter(
    __global const float* restrict imGlobal,
    __global float* restrict cellDescriptorGlobal,
    __local float* restrict imLocal,
    __local float* restrict derivX,
    __local float* restrict derivY,
    __local float* restrict cellDescriptorLocal,
    const int cellSize,
    const int binsPerCell,
    const int iteration,
    const int iterationsCount)
{
    const int2 wiId = (int2)(get_local_id(0), get_local_id(1));
    const int2 wgSize = (int2)(get_local_size(0), get_local_size(1));
    const int halfCellSize = cellSize / 2;
    const int2 imGlobalSize = (int2)(wgSize.x * iterationsCount, get_global_size(1));
    const int wiIdLinear = wiId.x + wiId.y * wgSize.x;

    // TODO reduce bank conflicts
    const int2 derivSize = wgSize + cellSize;
    const int2 imLocalSize = wgSize + (int2)(0, cellSize) + 2;

    const int2 imGlobalId = (int2)(
        min(wiId.x + halfCellSize + 1 + iteration * wgSize.x, imGlobalSize.x - 1),
        get_global_id(1));
    const int2 imLocalId = wiId + (int2)(1, halfCellSize + 1);
    const int2 derivId = wiId + (int2)(cellSize, halfCellSize);

    imLocal[imLocalId.x + 1 + imLocalId.y * imLocalSize.x] =
        imGlobal[imGlobalId.x + imGlobalId.y * imGlobalSize.x];
    if (wiId.y < halfCellSize + 1)
    {
        imLocal[imLocalId.x + 1 + (imLocalId.y - halfCellSize - 1) * imLocalSize.x] =
            imGlobal[imGlobalId.x + max(imGlobalId.y - halfCellSize - 1, 0) * imGlobalSize.x];
    }
    if (wiId.y >= wgSize.y - halfCellSize - 1)
    {
        imLocal[imLocalId.x + 1 + (imLocalId.y + halfCellSize + 1) * imLocalSize.x] =
            imGlobal[imGlobalId.x + imGlobalSize.y *
                min(imGlobalId.y + halfCellSize + 1, imGlobalSize.y - 1)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    calcPartialDerivs2(imLocal, derivX, derivY, derivId, imLocalId, derivSize.x, imLocalSize.x);
    if (wiId.y < halfCellSize)
    {
        calcPartialDerivs2(imLocal, derivX, derivY,
            derivId - (int2)(0, halfCellSize),
            imLocalId - (int2)(0, halfCellSize),
            derivSize.x, imLocalSize.x);
    }
    if (wiId.y >= wgSize.y - halfCellSize)
    {
        calcPartialDerivs2(imLocal, derivX, derivY,
            derivId + (int2)(0, halfCellSize),
            imLocalId + (int2)(0, halfCellSize),
            derivSize.x, imLocalSize.x);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int2 cellsCountLocal = wgSize / cellSize;
    const int binsCountLocal = cellsCountLocal.x * cellsCountLocal.y * binsPerCell;
    if (wiIdLinear < binsCountLocal)
    {
        cellDescriptorLocal[wiIdLinear] = 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    interpHog(derivX, derivY, cellDescriptorLocal, cellSize, binsPerCell);
    // TODO should we calculate insensitive bins in this kernel too?
    barrier(CLK_LOCAL_MEM_FENCE);

    const int cellIdLocalLin = wiIdLinear / binsPerCell;
    const int2 binIdLocal = wiId % binsPerCell;
    const int2 cellIdGlobal =
        (int2)(iteration, get_group_id(1)) * cellsCountLocal +
        (int2)(cellIdLocalLin % cellsCountLocal.x, cellIdLocalLin / cellsCountLocal.x);
    const int2 cellCountGlobal = imGlobalSize / cellSize;
    if (wiIdLinear < binsCountLocal)
    {
        cellDescriptorGlobal[(cellIdGlobal.x + cellIdGlobal.y) * binsPerCell] =
            cellDescriptorLocal[wiIdLinear];
    }
}

__kernel void calculateCellDescriptor(
    __global const float* restrict imGlobal,
    __global float* restrict cellDescriptorGlobal,
    __local float* restrict imLocal,
    __local float* restrict derivX,
    __local float* restrict derivY,
    __local float* restrict cellDescriptorLocal,
    const int iterationsCount,
    const int cellSize,
    const int binsPerCell)
{
    const int2 wgSize = (int2)(get_local_size(0), get_local_size(1));
    const int2 imLocalSize = wgSize + (int2)(0, cellSize) + 2;
    const int2 derivSize = wgSize + cellSize;
    const int2 wiId = (int2)(get_local_id(0), get_local_id(1));
    const int wiIdLinear = wiId.x + wiId.y * wgSize.x;

    // Load data and calculate partial derivatives for the very first image block. This
    // block differs from anothers because the leftmost part of the global image does not
    // exist.
    performHogInit(imGlobal, imLocal, derivX, derivY, cellSize, iterationsCount);

    for (int iteration = 0; iteration < iterationsCount - 1; ++iteration)
    {
        // Here we assume that the leftmost parts of the both local image and partial
        // derivatives have been already loaded, so we have to load the rest of data,
        // calculate the rest of derivatives and finally calculate the descriptor for
        // all the cells which are covered by current iteration.
        barrier(CLK_LOCAL_MEM_FENCE);
        performHogIter(imGlobal, cellDescriptorGlobal, imLocal, derivX, derivY,
            cellDescriptorLocal, cellSize, binsPerCell, iteration, iterationsCount);

        // Prepare to the next iteration: copy parts of local image and its derivatives.
        barrier(CLK_LOCAL_MEM_FENCE);
        const int2 derivIdCopy = (int2)(wiIdLinear % cellSize, wiIdLinear / cellSize);
        if (derivIdCopy.x < cellSize && derivIdCopy.y < derivSize.y)
        {
            const int i = derivIdCopy.x + derivIdCopy.y * derivSize.x;
            derivX[i] = derivX[i + wgSize.x];
            derivY[i] = derivY[i + wgSize.x];
        }
        const int2 imLocalIdCopy = (int2)(wiIdLinear % 2, wiIdLinear / 2);
        if (imLocalIdCopy.x < 2 && imLocalIdCopy.y < imLocalSize.y)
        {
            const int i = imLocalIdCopy.x + imLocalIdCopy.y * imLocalSize.x;
            imLocal[i] = imLocal[i + wgSize.x];
        }
    }

    // Process the last image block. Further copying of local data is not needed.
    barrier(CLK_LOCAL_MEM_FENCE);
    performHogIter(imGlobal, cellDescriptorGlobal, imLocal, derivX, derivY, cellDescriptorLocal,
        cellSize, binsPerCell, iterationsCount - 1, iterationsCount);
}

