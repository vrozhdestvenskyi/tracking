#ifndef RANGEDKERNEL_H
#define RANGEDKERNEL_H

#include <CL/cl.h>

struct RangedKernel
{
    void release();
    cl_int calculate(
        cl_command_queue queue,
        cl_int numWaitEvents,
        const cl_event *waitList,
        cl_event &event);

    cl_kernel kernel_ = NULL;
    cl_uint dim_ = 0;
    size_t ndrangeLoc_[3] = { 0, 0, 0 };
    size_t ndrangeGlob_[3] = { 0, 0, 0 };
};

#endif // RANGEDKERNEL_H
