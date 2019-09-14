#ifndef COLORCONVERSIONS_H
#define COLORCONVERSIONS_H

#include <rangedkernel.h>

enum class ColorConversion : int
{
    rgb2lab = 0,
    lab2rgb
};

struct Lab
{
    ~Lab();
    cl_int initialize(
        int width,
        int height,
        ColorConversion type,
        cl_context context,
        cl_program program,
        cl_mem image);
    void release();
    cl_int calculate(
        cl_command_queue queue,
        cl_int numWaitEvents,
        const cl_event *waitList,
        cl_event &event);

    cl_mem converted_ = NULL;
    RangedKernel kernel_;
};

#endif // COLORCONVERSIONS_H
