__kernel void calculatePartialDerivatives(
    __global const float* restrict image,
    __global float* restrict derivativesX,
    __global float* restrict derivativesY,
    __local float* restrict imageLocal,
    const int iterations)
{
    const int widthGlobal = get_local_size(0) * iterations;
    const int heightGlobal = get_global_size(1);
    const int widthLocal = get_local_size(0) + 2;
    const int heightLocal = get_local_size(1) + 2;
    const int xLocal = get_local_id(0) + 1;
    const int yLocal = get_local_id(1) + 1;
    const int yGlobal = get_global_id(1);
    int xGlobal = get_local_id(0);

    imageLocal[xLocal + yLocal * widthLocal] = image[xGlobal + yGlobal * widthGlobal];
    if (xLocal == 1)
    {
        imageLocal[yLocal * widthLocal] = image[max(0, xGlobal - 1) + yGlobal * widthGlobal];
    }
    if (xLocal == widthLocal - 2)
    {
        imageLocal[xLocal + 1 + yLocal * widthLocal] =
            image[min(widthGlobal - 1, xGlobal + 1) + yGlobal * widthGlobal];
    }
    if (yLocal == 1)
    {
        imageLocal[xLocal] = image[xGlobal + max(0, yGlobal - 1) * widthGlobal];
    }
    if (yLocal == heightLocal - 2)
    {
        imageLocal[xLocal + (yLocal + 1) * widthLocal] =
            image[xGlobal + min(heightGlobal - 1, yGlobal + 1) * widthGlobal];
    }

    for (int iteration = 0; iteration + 1 < iterations; ++iteration)
    {
        barrier(CLK_LOCAL_MEM_FENCE);

        derivativesX[xGlobal + yGlobal * widthGlobal] = 0.5f * (
            imageLocal[xLocal + 1 + yLocal * widthLocal] -
            imageLocal[xLocal - 1 + yLocal * widthLocal]);
        derivativesY[xGlobal + yGlobal * widthGlobal] = 0.5f * (
            imageLocal[xLocal + (yLocal + 1) * widthLocal] -
            imageLocal[xLocal + (yLocal - 1) * widthLocal]);

        barrier(CLK_LOCAL_MEM_FENCE);

        if (xLocal < 3)
        {
            imageLocal[xLocal - 1 + yLocal * widthLocal] =
                imageLocal[widthLocal - 3 + xLocal + yLocal * widthLocal];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        xGlobal += get_local_size(0);
        const int xLocalSrc = xLocal + 1;
        const int xGlobalSrc = min(widthGlobal - 1, xGlobal + 1);

        imageLocal[xLocalSrc + yLocal * widthLocal] = image[xGlobalSrc + yGlobal * widthGlobal];
        if (yLocal == 1)
        {
            imageLocal[xLocalSrc] = image[xGlobalSrc + max(0, yGlobal - 1) * widthGlobal];
        }
        if (yLocal == heightLocal - 2)
        {
            imageLocal[xLocalSrc + (yLocal + 1) * widthLocal] =
                image[xGlobalSrc + min(heightGlobal - 1, yGlobal + 1) * widthGlobal];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    derivativesX[xGlobal + yGlobal * widthGlobal] = 0.5f * (
        imageLocal[xLocal + 1 + yLocal * widthLocal] -
        imageLocal[xLocal - 1 + yLocal * widthLocal]);
    derivativesY[xGlobal + yGlobal * widthGlobal] = 0.5f * (
        imageLocal[xLocal + (yLocal + 1) * widthLocal] -
        imageLocal[xLocal + (yLocal - 1) * widthLocal]);
}

