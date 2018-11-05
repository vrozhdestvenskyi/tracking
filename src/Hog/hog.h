#ifndef HOG_H
#define HOG_H

#include <array>
#include <CL/cl.h>
#include <hogproto.h>

class Hog
{
public:
    ~Hog();
    cl_int initialize(const HogSettings &settings, cl_context context, cl_program program);
    void release();
    cl_int calculate(
        cl_command_queue commandQueue,
        cl_int numWaitEvents,
        const cl_event *waitList,
        cl_event &event);

    cl_mem image_ = NULL; // TODO remove this
    cl_mem cellDescriptor_ = NULL;
    cl_kernel kernel_ = NULL;
    size_t ndrangeLocal_[2] = { 0, 0 };
    size_t ndrangeGlobal_[2] = { 0, 0 };
};

#endif // HOG_H

