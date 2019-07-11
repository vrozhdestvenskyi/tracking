#ifndef HOG_H
#define HOG_H

#include <array>
#include <hogproto.h>
#include <rangedkernel.h>

class CellHog
{
public:
    ~CellHog();
    cl_int initialize(
        const HogSettings &settings,
        cl_context context,
        cl_program program,
        cl_mem image);
    void release();
    cl_int calculate(
        cl_command_queue queue,
        cl_int numWaitEvents,
        const cl_event *waitList,
        cl_event &event);

    cl_mem descriptor_ = NULL;
    RangedKernel kernel_;
};

class CellNorm
{
public:
    ~CellNorm();
    cl_int initialize(
        const HogSettings &settings,
        cl_context context,
        cl_program program,
        cl_mem sensitiveCellDescriptor);
    void release();
    cl_int calculate(
        cl_command_queue queue,
        cl_int numWaitEvents,
        const cl_event *waitList,
        cl_event &event);

    cl_int2 padding_ = cl_int2{0, 0};
    cl_mem cellNorms_ = NULL;
    RangedKernel kernel_;
};

class InvBlockNorm
{
public:
    ~InvBlockNorm();
    cl_int initialize(
        const HogSettings &settings,
        const cl_int2 &padding,
        cl_context context,
        cl_program program,
        cl_mem cellNorms);
    void release();
    cl_int calculate(
        cl_command_queue queue,
        cl_int numWaitEvents,
        const cl_event *waitList,
        cl_event &event);

    cl_mem invBlockNorms_ = NULL;
    RangedKernel kernel_;
};

class BlockHog
{
public:
    ~BlockHog();
    cl_int initialize(
        const HogSettings &settings,
        const cl_int2 &padding,
        cl_context context,
        cl_program program,
        cl_mem cellDesc,
        cl_mem invBlockNorms);
    void release();
    cl_int calculate(
        cl_command_queue queue,
        cl_int numWaitEvents,
        const cl_event *waitList,
        cl_event &event);

    cl_mem descriptor_ = NULL;
    RangedKernel kernel_;
};

class Hog
{
public:
    ~Hog();
    cl_int initialize(
        const HogSettings &settings,
        cl_context context,
        cl_program program,
        cl_mem image);
    void release();
    cl_int calculate(
        cl_command_queue queue,
        cl_int numWaitEvents,
        const cl_event *waitList,
        cl_event &event);

    CellHog cellHog_;
    CellNorm cellNorm_;
    InvBlockNorm invBlockNorm_;
    BlockHog blockHog_;
};

#endif // HOG_H

