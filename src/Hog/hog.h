#ifndef HOG_H
#define HOG_H

typedef unsigned char uchar;

// TODO: update HogWidget by insensitiveBinCount_; use these settings in Piotr's method
struct HogSettings
{
    int sensitiveBinCount() const { return insensitiveBinCount_ * 2; }
    int channelsPerCell() const { return insensitiveBinCount_ + sensitiveBinCount(); }
    int channelsPerFeature() const { return channelsPerCell() + 4; }

    int insensitiveBinCount_ = 0; // 9
    int cellSize_ = 0; // 4
    int cellCount_[2] = { 0, 0 };
    float truncation_ = 0.0f; // 0.2f
};

struct HogHandle
{
    HogSettings settings_;
    float *cellSquaredNorms_ = nullptr;
    float *blockInverseNorms_ = nullptr;
    float *cellDescriptor_ = nullptr;
    float *featureDescriptor_ = nullptr;
    float *cellInterpWeights_ = nullptr;
};

void initializeHogHandle(const HogSettings &settings, HogHandle &handle);
void releaseHogHandle(HogHandle &handle);
void calculateHog(const uchar *image, HogHandle &handle);

#endif // HOG_H
