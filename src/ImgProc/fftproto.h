#ifndef FFTPROTO_H
#define FFTPROTO_H

#include <vector>

using uint = unsigned int;

class FftProto
{
public:
    ~FftProto();
    bool init(const uint N, const std::vector<uint> *stages = nullptr);
    void release();
    bool calcForward(const float *srcReal);
    bool calc(const float *srcComplex, const bool inverse);
    const float* result() const;

protected:
    bool initRadixStages();
    void initDigitReversal();
    bool calcRadixStages(const bool inverse);

    uint N_ = 0U;
    std::vector<uint> stages_;
    uint *digitReversal_ = nullptr;
    float *dstComplex_ = nullptr;
};

class Fft2dProto
{
public:
    ~Fft2dProto();
    bool init(const uint width, const uint height);
    void release();
    bool calcForward(const float *srcReal);
    bool calc(const float *srcComplex, const bool inverse);
    const float *result() const;

protected:
    bool calc(const bool inverse);

    uint width_ = 0U;
    uint height_ = 0U;
    FftProto hor_;
    FftProto ver_;
    float *dstComplex_ = nullptr;
    float *transposed_ = nullptr;
};

#endif // FFTPROTO_H

