#ifndef HOGPROTO_H
#define HOGPROTO_H

typedef unsigned char uchar;

// TODO: use these settings in Piotr's method.
struct HogSettings
{
    bool init(int imWidth, int imHeight);

    static int sensitiveBinCount() { return insensitiveBinCount_ * 2; }
    static int channelsPerCell() { return insensitiveBinCount_ + sensitiveBinCount(); }
    static int channelsPerBlock() { return channelsPerCell() + 4; }

    int imWidth() const;
    int imHeight() const;

    static const int insensitiveBinCount_ = 9;
    static const int cellSize_ = 4;
    static constexpr const int wgSize_[2] = { 16, 16 };
    static constexpr const float truncation_ = 0.2f;

    int cellCount_[2] = { 0, 0 };
};

class HogProto
{
public:
    ~HogProto();
    void initialize(const HogSettings &settings);
    void release();
    void calculate(const float *image);

    HogSettings settings_;
    float *cellSquaredNorms_ = nullptr;
    float *blockInverseNorms_ = nullptr;
    float *cellDescriptor_ = nullptr;
    float *blockDescriptor_ = nullptr;
    float *cellInterpWeights_ = nullptr;

protected:
    void calculateCellInterpolationWeights();
    void calculateCellDescriptor(const float *image);
    void calculateInsensitiveNorms();
    void applyNormalization();
};

#endif // HOGPROTO_H
