#include <hog.h>

Hog::~Hog()
{
    release();
}

cl_int Hog::initialize(const HogSettings &settings)
{
    settings_ = settings;
    return CL_SUCCESS;
}

void Hog::release()
{}

cl_int Hog::calculate(const uchar *image)
{
    return CL_SUCCESS;
}
