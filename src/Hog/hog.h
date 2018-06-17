#ifndef HOG_H
#define HOG_H

#include <hogproto.h>

class Hog
{
public:
    ~Hog();
    void initialize(const HogSettings &settings);
    void release();
    void calculate(const uchar *image);

    HogSettings settings_;
};

#endif // HOG_H

