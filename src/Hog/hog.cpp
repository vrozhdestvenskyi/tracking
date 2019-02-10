#include <hog.h>
#include <vector>

void RangedKernel::release()
{
    if (kernel_)
    {
        clReleaseKernel(kernel_);
        kernel_ = NULL;
    }
}

cl_int RangedKernel::calculate(
    cl_command_queue commandQueue,
    cl_int numWaitEvents,
    const cl_event *waitList,
    cl_event &event)
{
    return clEnqueueNDRangeKernel(commandQueue, kernel_, dim_, NULL,
        ndrangeGlob_, ndrangeLoc_, numWaitEvents, waitList, &event);
}

Derivs::~Derivs()
{
    release();
}

cl_int Derivs::initialize(
    const HogSettings &settings,
    cl_context context,
    cl_program program,
    cl_mem image)
{
    kernel_.dim_ = 2;
    for (int i = 0; i < 2; ++i)
    {
        kernel_.ndrangeLoc_[i] = settings.wgSize_[i];
    }
    kernel_.ndrangeGlob_[0] = settings.imWidth();
    kernel_.ndrangeGlob_[1] = kernel_.ndrangeLoc_[1];
    if (kernel_.ndrangeGlob_[0] % kernel_.ndrangeLoc_[0] ||
        settings.imHeight() % kernel_.ndrangeLoc_[1])
    {
        return CL_INVALID_WORK_GROUP_SIZE;
    }

    size_t bytes = (settings.imWidth() + settings.cellSize_) *
        (settings.imHeight() + settings.cellSize_) * sizeof(cl_float);
    {
        std::vector<float> zeros(bytes, 0.0f);
        derivsX_ = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            bytes, zeros.data(), NULL);
        if (derivsX_)
        {
            derivsY_ = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                bytes, zeros.data(), NULL);
        }
    }
    if (derivsY_)
    {
        kernel_.kernel_ = clCreateKernel(program, "calcDerivs", NULL);
    }
    if (!kernel_.kernel_)
    {
        release();
        return CL_INVALID_KERNEL;
    }

    int argId = 0;
    cl_int status = clSetKernelArg(kernel_.kernel_, argId++, sizeof(cl_mem), &image);
    status |= clSetKernelArg(kernel_.kernel_, argId++, sizeof(cl_mem), &derivsX_);
    status |= clSetKernelArg(kernel_.kernel_, argId++, sizeof(cl_mem), &derivsY_);
    int iterationsCount = settings.imHeight() / kernel_.ndrangeLoc_[1];
    status |= clSetKernelArg(kernel_.kernel_, argId++, sizeof(cl_int), &iterationsCount);
    int halfPad = settings.cellSize_ / 2;
    status |= clSetKernelArg(kernel_.kernel_, argId++, sizeof(cl_int), &halfPad);
    return status;
}

void Derivs::release()
{
    kernel_.release();
    if (derivsY_)
    {
        clReleaseMemObject(derivsY_);
        derivsY_ = NULL;
    }
    if (derivsX_)
    {
        clReleaseMemObject(derivsX_);
        derivsX_ = NULL;
    }
}

cl_int Derivs::calculate(
    cl_command_queue commandQueue,
    cl_int numWaitEvents,
    const cl_event *waitList,
    cl_event &event)
{
    return kernel_.calculate(commandQueue, numWaitEvents, waitList, event);
}

CellHog::~CellHog()
{
    release();
}

cl_int CellHog::initialize(
    const HogSettings &settings,
    cl_context context,
    cl_program program,
    cl_mem derivsX,
    cl_mem derivsY)
{
    kernel_.dim_ = 2;
    for (int i = 0; i < 2; ++i)
    {
        kernel_.ndrangeLoc_[i] = settings.wgSize_[i];
    }
    kernel_.ndrangeGlob_[0] = settings.imWidth();
    kernel_.ndrangeGlob_[1] = kernel_.ndrangeLoc_[1];
    if (kernel_.ndrangeGlob_[0] % kernel_.ndrangeLoc_[0] ||
        settings.imHeight() % kernel_.ndrangeLoc_[1])
    {
        return CL_INVALID_WORK_GROUP_SIZE;
    }

    int bytes = settings.cellCount_[0] * settings.cellCount_[1] * settings.sensitiveBinCount() *
        sizeof(cl_uint);
    descriptor_ = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, NULL);
    if (descriptor_)
    {
        kernel_.kernel_ = clCreateKernel(program, "calcCellDesc", NULL);
    }
    if (!kernel_.kernel_)
    {
        release();
        return CL_INVALID_KERNEL;
    }

    int argId = 0;
    cl_int status = clSetKernelArg(kernel_.kernel_, argId++, sizeof(cl_mem), &derivsX);
    status |= clSetKernelArg(kernel_.kernel_, argId++, sizeof(cl_mem), &derivsY);
    status |= clSetKernelArg(kernel_.kernel_, argId++, sizeof(cl_mem), &descriptor_);
    int iterationsCount = settings.imHeight() / kernel_.ndrangeLoc_[1];
    status |= clSetKernelArg(kernel_.kernel_, argId++, sizeof(cl_int), &iterationsCount);
    return status;
}

void CellHog::release()
{
    kernel_.release();
    if (descriptor_)
    {
        clReleaseMemObject(descriptor_);
        descriptor_ = NULL;
    }
}

cl_int CellHog::calculate(
    cl_command_queue commandQueue,
    cl_int numWaitEvents,
    const cl_event *waitList,
    cl_event &event)
{
    return kernel_.calculate(commandQueue, numWaitEvents, waitList, event);
}

CellNorm::~CellNorm()
{
    release();
}

cl_int CellNorm::initialize(
    const HogSettings &settings,
    cl_context context,
    cl_program program,
    cl_mem sensitiveCellDescriptor)
{
    kernel_.dim_ = 2;
    kernel_.ndrangeLoc_[0] = kernel_.ndrangeLoc_[1] = 4;
    if (settings.cellCount_[0] % kernel_.ndrangeLoc_[0] ||
        settings.cellCount_[1] % kernel_.ndrangeLoc_[1])
    {
        return CL_INVALID_WORK_GROUP_SIZE;
    }
    kernel_.ndrangeGlob_[0] = settings.cellCount_[0];
    kernel_.ndrangeGlob_[1] = kernel_.ndrangeLoc_[1];
    padding_ = { (int)kernel_.ndrangeLoc_[0] + 1, (int)kernel_.ndrangeLoc_[1] + 1 };

    size_t bytes = (settings.cellCount_[0] + padding_.x) * (settings.cellCount_[1] + padding_.y) *
        sizeof(cl_float);
    {
        std::vector<float> zeros(bytes, 0.0f);
        cellNorms_ = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            bytes, zeros.data(), NULL);
    }
    if (cellNorms_)
    {
        kernel_.kernel_ = clCreateKernel(program, "calcCellNorms", NULL);
    }
    if (!kernel_.kernel_)
    {
        release();
        return CL_INVALID_KERNEL;
    }

    int argId = 0;
    cl_int status = clSetKernelArg(kernel_.kernel_, argId++, sizeof(cl_mem), &sensitiveCellDescriptor);
    status |= clSetKernelArg(kernel_.kernel_, argId++, sizeof(cl_mem), &cellNorms_);
    int iterationsCount = settings.cellCount_[1] / kernel_.ndrangeLoc_[1];
    status |= clSetKernelArg(kernel_.kernel_, argId++, sizeof(cl_int), &iterationsCount);
    status |= clSetKernelArg(kernel_.kernel_, argId++, sizeof(cl_int), &padding_.x);
    return status;
}

void CellNorm::release()
{
    kernel_.release();
    if (cellNorms_)
    {
        clReleaseMemObject(cellNorms_);
        cellNorms_ = NULL;
    }
}

cl_int CellNorm::calculate(
    cl_command_queue commandQueue,
    cl_int numWaitEvents,
    const cl_event *waitList,
    cl_event &event)
{
    return kernel_.calculate(commandQueue, numWaitEvents, waitList, event);
}

CellNormSumX::~CellNormSumX()
{
    release();
}

cl_int CellNormSumX::initialize(
    const HogSettings &settings,
    const cl_int2 &padding,
    cl_context context,
    cl_program program,
    cl_mem cellNorms)
{
    kernel_.dim_ = 2;
    kernel_.ndrangeLoc_[0] = kernel_.ndrangeLoc_[1] = 4;
    kernel_.ndrangeGlob_[0] = kernel_.ndrangeLoc_[0];
    kernel_.ndrangeGlob_[1] = settings.cellCount_[1] + padding.y - 1;
    if ((settings.cellCount_[0] + padding.x - 1) % kernel_.ndrangeLoc_[0] ||
        kernel_.ndrangeGlob_[1] % kernel_.ndrangeLoc_[1])
    {
        return CL_INVALID_WORK_GROUP_SIZE;
    }

    size_t bytes = (settings.cellCount_[0] + padding.x) * (settings.cellCount_[1] + padding.y) *
        sizeof(cl_float);
    {
        std::vector<float> zeros(bytes, 0.0f);
        normSums_ = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            bytes, zeros.data(), NULL);
    }
    if (normSums_)
    {
        kernel_.kernel_ = clCreateKernel(program, "sumCellNormsX", NULL);
    }
    if (!kernel_.kernel_)
    {
        release();
        return CL_INVALID_KERNEL;
    }

    int argId = 0;
    cl_int status = clSetKernelArg(kernel_.kernel_, argId++, sizeof(cl_mem), &cellNorms);
    status |= clSetKernelArg(kernel_.kernel_, argId++, sizeof(cl_mem), &normSums_);
    int iterationsCount = (settings.cellCount_[0] + padding.x - 1) / kernel_.ndrangeLoc_[0];
    status |= clSetKernelArg(kernel_.kernel_, argId++, sizeof(cl_int), &iterationsCount);
    return status;
}

void CellNormSumX::release()
{
    kernel_.release();
    if (normSums_)
    {
        clReleaseMemObject(normSums_);
        normSums_ = NULL;
    }
}

cl_int CellNormSumX::calculate(
    cl_command_queue commandQueue,
    cl_int numWaitEvents,
    const cl_event *waitList,
    cl_event &event)
{
    return kernel_.calculate(commandQueue, numWaitEvents, waitList, event);
}

InvBlockNorm::~InvBlockNorm()
{
    release();
}

cl_int InvBlockNorm::initialize(
    const HogSettings &settings,
    const cl_int2 &padding,
    cl_context context,
    cl_program program,
    cl_mem cellNorms)
{
    kernel_.dim_ = 2;
    kernel_.ndrangeLoc_[0] = kernel_.ndrangeLoc_[1] = 4;
    kernel_.ndrangeGlob_[0] = settings.cellCount_[0] + padding.x - 1;
    kernel_.ndrangeGlob_[1] = kernel_.ndrangeLoc_[1];
    if (kernel_.ndrangeGlob_[0] % kernel_.ndrangeLoc_[0] ||
        (settings.cellCount_[1] + padding.y - 1) % kernel_.ndrangeLoc_[1])
    {
        return CL_INVALID_WORK_GROUP_SIZE;
    }

    size_t bytes = (settings.cellCount_[0] + padding.x) * (settings.cellCount_[1] + padding.y) *
        sizeof(cl_float);
    {
        std::vector<float> zeros(bytes, 0.0f);
        invBlockNorms_ = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            bytes, zeros.data(), NULL);
    }
    if (invBlockNorms_)
    {
        kernel_.kernel_ = clCreateKernel(program, "calcInvBlockNorms", NULL);
    }
    if (!kernel_.kernel_)
    {
        release();
        return CL_INVALID_KERNEL;
    }

    int argId = 0;
    cl_int status = clSetKernelArg(kernel_.kernel_, argId++, sizeof(cl_mem), &cellNorms);
    status |= clSetKernelArg(kernel_.kernel_, argId++, sizeof(cl_mem), &invBlockNorms_);
    int iterationsCount = (settings.cellCount_[1] + padding.y - 1) / kernel_.ndrangeLoc_[1];
    status |= clSetKernelArg(kernel_.kernel_, argId++, sizeof(cl_int), &iterationsCount);
    return status;
}

void InvBlockNorm::release()
{
    kernel_.release();
    if (invBlockNorms_)
    {
        clReleaseMemObject(invBlockNorms_);
        invBlockNorms_ = NULL;
    }
}

cl_int InvBlockNorm::calculate(
    cl_command_queue commandQueue,
    cl_int numWaitEvents,
    const cl_event *waitList,
    cl_event &event)
{
    return kernel_.calculate(commandQueue, numWaitEvents, waitList, event);
}

BlockHog::~BlockHog()
{
    release();
}

cl_int BlockHog::initialize(
    const HogSettings &settings,
    const cl_int2 &padding,
    cl_context context,
    cl_program program,
    cl_mem cellDesc,
    cl_mem invBlockNorms)
{
    kernel_.dim_ = 2;
    kernel_.ndrangeLoc_[0] = kernel_.ndrangeLoc_[1] = 4;
    kernel_.ndrangeGlob_[0] = settings.cellCount_[0];
    kernel_.ndrangeGlob_[1] = kernel_.ndrangeLoc_[1];
    if (kernel_.ndrangeGlob_[0] % kernel_.ndrangeLoc_[0] ||
        settings.cellCount_[1] % kernel_.ndrangeLoc_[1])
    {
        return CL_INVALID_WORK_GROUP_SIZE;
    }

    size_t bytes = settings.cellCount_[0] * settings.cellCount_[1] * settings.channelsPerBlock() *
        sizeof(cl_float);
    descriptor_ = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, NULL);
    if (descriptor_)
    {
        kernel_.kernel_ = clCreateKernel(program, "applyNormalization", NULL);
    }
    if (!kernel_.kernel_)
    {
        release();
        return CL_INVALID_KERNEL;
    }

    int argId = 0;
    cl_int status = clSetKernelArg(kernel_.kernel_, argId++, sizeof(cl_mem), &cellDesc);
    status |= clSetKernelArg(kernel_.kernel_, argId++, sizeof(cl_mem), &invBlockNorms);
    status |= clSetKernelArg(kernel_.kernel_, argId++, sizeof(cl_mem), &descriptor_);
    int iterationsCount = settings.cellCount_[1] / kernel_.ndrangeLoc_[1];
    status |= clSetKernelArg(kernel_.kernel_, argId++, sizeof(cl_int), &iterationsCount);
    status |= clSetKernelArg(kernel_.kernel_, argId++, sizeof(cl_int), &padding.x);
    return status;
}

void BlockHog::release()
{
    kernel_.release();
    if (descriptor_)
    {
        clReleaseMemObject(descriptor_);
        descriptor_ = NULL;
    }
}

cl_int BlockHog::calculate(
    cl_command_queue commandQueue,
    cl_int numWaitEvents,
    const cl_event *waitList,
    cl_event &event)
{
    return kernel_.calculate(commandQueue, numWaitEvents, waitList, event);
}

