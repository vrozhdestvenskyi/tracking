#include <colorconversions.h>
#include <string>

Lab::~Lab()
{
    release();
}

cl_int Lab::initialize(
    int width,
    int height,
    ColorConversion type,
    cl_context context,
    cl_program program,
    cl_mem image)
{
    kernel_.dim_ = 2;
    kernel_.ndrangeLoc_[0] = kernel_.ndrangeLoc_[1] = 16;
    kernel_.ndrangeGlob_[0] = width;
    kernel_.ndrangeGlob_[1] = kernel_.ndrangeLoc_[1];
    if (kernel_.ndrangeGlob_[0] % kernel_.ndrangeLoc_[0] || height % kernel_.ndrangeLoc_[1])
    {
        return CL_INVALID_WORK_GROUP_SIZE;
    }

    converted_ = clCreateBuffer(context, CL_MEM_READ_WRITE,
        width * height * 3 * sizeof(cl_uchar), NULL, NULL);
    if (converted_)
    {
        std::string name;
        name = type == ColorConversion::lab2rgb ? "lab2rgb" : name;
        name = type == ColorConversion::rgb2lab ? "rgb2lab" : name;
        kernel_.kernel_ = clCreateKernel(program, name.c_str(), NULL);
    }
    if (!kernel_.kernel_)
    {
        release();
        return CL_INVALID_KERNEL;
    }

    int argId = 0;
    cl_int status = clSetKernelArg(kernel_.kernel_, argId++, sizeof(cl_mem), &image);
    status |= clSetKernelArg(kernel_.kernel_, argId++, sizeof(cl_mem), &converted_);
    int iterationsCount = height / kernel_.ndrangeLoc_[1];
    status |= clSetKernelArg(kernel_.kernel_, argId++, sizeof(cl_int), &iterationsCount);
    return status;
}

void Lab::release()
{
    kernel_.release();
    if (converted_)
    {
        clReleaseMemObject(converted_);
        converted_ = NULL;
    }
}

cl_int Lab::calculate(
    cl_command_queue queue,
    cl_int numWaitEvents,
    const cl_event *waitList,
    cl_event &event)
{
    return kernel_.calculate(queue, numWaitEvents, waitList, event);
}

