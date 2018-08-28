#include <hog.h>

Hog::~Hog()
{
    release();
}

cl_int Hog::initialize(const HogSettings &settings, cl_context context, cl_program program)
{
    ndrangeLocal_[0] = ndrangeLocal_[1] = 16;
    const int imageSize[2] = {
        settings.cellCount_[0] * settings.cellSize_,
        settings.cellCount_[1] * settings.cellSize_ };
    if (imageSize[0] % ndrangeLocal_[0] || imageSize[1] % ndrangeLocal_[1])
    {
        return CL_INVALID_BUFFER_SIZE;
    }
    ndrangeGlobal_[0] = ndrangeLocal_[0];
    ndrangeGlobal_[1] = imageSize[1];

    int bytes = imageSize[0] * imageSize[1] * sizeof(cl_float);
    image_ = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    if (image_)
    {
        derivativesX_ = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, NULL);
    }
    if (derivativesX_)
    {
        derivativesY_ = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, NULL);
    }
    if (derivativesY_)
    {
        kernel_ = clCreateKernel(program, "calculatePartialDerivatives", NULL);
    }
    if (!kernel_)
    {
        release();
        return CL_INVALID_KERNEL;
    }

    int argId = 0;
    cl_int status = clSetKernelArg(kernel_, argId++, sizeof(cl_mem), &image_);
    status |= clSetKernelArg(kernel_, argId++, sizeof(cl_mem), &derivativesX_);
    status |= clSetKernelArg(kernel_, argId++, sizeof(cl_mem), &derivativesY_);
    bytes = (ndrangeLocal_[0] + 2) * (ndrangeLocal_[1] + 2) * sizeof(cl_float);
    status |= clSetKernelArg(kernel_, argId++, bytes, NULL);
    int blocksX = imageSize[0] / ndrangeLocal_[0];
    status |= clSetKernelArg(kernel_, argId++, sizeof(cl_int), &blocksX);
    return status;
}

void Hog::release()
{
    if (kernel_)
    {
        clReleaseKernel(kernel_);
        kernel_ = nullptr;
    }
    if (derivativesY_)
    {
        clReleaseMemObject(derivativesY_);
        derivativesY_ = NULL;
    }
    if (derivativesX_)
    {
        clReleaseMemObject(derivativesX_);
        derivativesX_ = NULL;
    }
    if (image_)
    {
        clReleaseMemObject(image_);
        image_ = NULL;
    }
}

cl_int Hog::calculate(
    cl_command_queue commandQueue,
    cl_int numWaitEvents,
    const cl_event *waitList,
    cl_event &event)
{
    return clEnqueueNDRangeKernel(commandQueue, kernel_, 2, NULL, ndrangeGlobal_,
        ndrangeLocal_, numWaitEvents, waitList, &event);
}

