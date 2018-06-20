#ifndef HOG_H
#define HOG_H

#include <CL/cl.h>
#include <hogproto.h>

class Hog
{
public:
    ~Hog();
    cl_int initialize(const HogSettings &settings);
    void release();
    cl_int calculate(const uchar *image);

    HogSettings settings_;
};

#endif // HOG_H

