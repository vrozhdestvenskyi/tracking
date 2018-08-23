#ifndef HOG_H
#define HOG_H

#include <array>
#include <CL/cl.h>
#include <hogproto.h>

class Hog
{
public:
    ~Hog();
    cl_int initialize(const HogSettings &settings);
    void release();
    cl_int calculate(const float *image);

    HogSettings settings_;
    int ndrangeLocal_[2];
    int ndrangeGlobal_[2];

};

#endif // HOG_H

