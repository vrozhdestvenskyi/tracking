#include <hog.h>

Hog::~Hog()
{
    release();
}

void Hog::initialize(const HogSettings &settings)
{
    settings_ = settings;
}
