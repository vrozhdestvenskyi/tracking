#include <hog.h>

Hog::~Hog()
{
    release();
}

cl_int Hog::initialize(const HogSettings &settings)
{
    settings_ = settings;
    ndrangeLocal_[0] = ndrangeLocal_[1] = 16;
    const int imageSize[2] = {
        settings.cellCount_[0] * settings.cellSize_,
        settings.cellCount_[1] * settings.cellSize_ };
    if (imageSize[0] % ndrangeLocal_[0] || imageSize[1] % ndrangeLocal_[1])
    {
        return CL_INVALID_BUFFER_SIZE;
    }
    ndrangeGlobal_[0] = ndrangeLocal_[0];
    ndrangeGlobal_[1] = imageSize[1] / ndrangeLocal_[1];
    return CL_SUCCESS;
}

void Hog::release()
{}

cl_int Hog::calculate(const float *image)
{
    return CL_SUCCESS;
}
