#include <rangedkernel.h>

void RangedKernel::release()
{
    if (kernel_)
    {
        clReleaseKernel(kernel_);
        kernel_ = NULL;
    }
}

cl_int RangedKernel::calculate(
    cl_command_queue queue,
    cl_int numWaitEvents,
    const cl_event *waitList,
    cl_event &event)
{
    return clEnqueueNDRangeKernel(queue, kernel_, dim_, NULL,
        ndrangeGlob_, ndrangeLoc_, numWaitEvents, waitList, &event);
}

