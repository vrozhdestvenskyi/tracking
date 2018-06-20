#ifndef HOGPROTO_H
#define HOGPROTO_H

typedef unsigned char uchar;

// TODO: use these settings in Piotr's method.
struct HogSettings
{
    int sensitiveBinCount() const { return insensitiveBinCount_ * 2; }
    int channelsPerCell() const { return insensitiveBinCount_ + sensitiveBinCount(); }
    int channelsPerFeature() const { return channelsPerCell() + 4; }

    // TODO: create a separate structure HogParameters
    int insensitiveBinCount_ = 9;
    int cellSize_ = 4;
    float truncation_ = 0.2f;

    int cellCount_[2] = { 0, 0 };
};

class HogProto
{
public:
    ~HogProto();
    void initialize(const HogSettings &settings);
    void release();
    void calculate(const uchar *image);

    HogSettings settings_;
    float *cellSquaredNorms_ = nullptr;
    float *blockInverseNorms_ = nullptr;
    float *cellDescriptor_ = nullptr;
    float *featureDescriptor_ = nullptr;
    float *cellInterpWeights_ = nullptr;

protected:
    void calculateCellInterpolationWeights();
    void calculateCellDescriptor(const uchar *image);
    void calculateInsensitiveNorms();
    void applyNormalization();
};

#endif // HOGPROTO_H
