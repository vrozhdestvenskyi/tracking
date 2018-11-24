#include <hog.h>

Hog::~Hog()
{
    release();
}

cl_int Hog::initialize(const HogSettings &settings, cl_context context, cl_program program)
{
    for (int i = 0; i < 2; ++i)
    {
        ndrangeLocal_[i] = settings.wgSize_[i];
    }
    ndrangeGlobal_[0] = ndrangeLocal_[0];
    ndrangeGlobal_[1] = settings.cellCount_[1] * settings.cellSize_;

    int bytes = settings.imSize_[0] * settings.imSize_[1] * sizeof(cl_float);
    image_ = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    bytes = settings.cellCount_[0] * settings.cellCount_[1] * settings.sensitiveBinCount() *
        sizeof(cl_uint);
    if (image_)
    {
        cellDescriptor_ = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, NULL);
    }
    if (cellDescriptor_)
    {
        kernel_ = clCreateKernel(program, "calculateCellDescriptor", NULL);
    }
    if (!kernel_)
    {
        release();
        return CL_INVALID_KERNEL;
    }

    int cellSize = settings.cellSize_;
    int binsPerCell = settings.sensitiveBinCount();
    cl_int2 halfPadding = { settings.halfPadding_[0], settings.halfPadding_[1] };

    int argId = 0;
    cl_int status = clSetKernelArg(kernel_, argId++, sizeof(cl_mem), &image_);
    status |= clSetKernelArg(kernel_, argId++, sizeof(cl_mem), &cellDescriptor_);
    bytes = (ndrangeLocal_[0] + 2) * (ndrangeLocal_[1] + 2 + cellSize) * sizeof(cl_float);
    status |= clSetKernelArg(kernel_, argId++, bytes, NULL);
    bytes = (ndrangeLocal_[0] + cellSize) * (ndrangeLocal_[1] + cellSize) * sizeof(cl_float);
    status |= clSetKernelArg(kernel_, argId++, bytes, NULL);
    status |= clSetKernelArg(kernel_, argId++, bytes, NULL);
    bytes = ndrangeLocal_[0] * ndrangeLocal_[1] * 2 * sizeof(cl_uint);
    status |= clSetKernelArg(kernel_, argId++, bytes, NULL);
    int iterationsCount = settings.cellCount_[0] * settings.cellSize_ / ndrangeLocal_[0];
    status |= clSetKernelArg(kernel_, argId++, sizeof(cl_int), &iterationsCount);
    status |= clSetKernelArg(kernel_, argId++, sizeof(cl_int), &cellSize);
    status |= clSetKernelArg(kernel_, argId++, sizeof(cl_int), &binsPerCell);
    status |= clSetKernelArg(kernel_, argId++, sizeof(cl_int2), &halfPadding);
    return status;
}

void Hog::release()
{
    if (kernel_)
    {
        clReleaseKernel(kernel_);
        kernel_ = nullptr;
    }
    if (cellDescriptor_)
    {
        clReleaseMemObject(cellDescriptor_);
        cellDescriptor_ = NULL;
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

