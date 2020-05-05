#ifndef FFTPROTO_H
#define FFTPROTO_H

using uint = unsigned int;

class FftProto
{
public:
    FftProto();
    ~FftProto();
    bool init(const uint N);
    void release();
    bool calcForward(const float *srcReal);
    bool calc(const float *srcComplex, const bool inverse);
    const float *result() const;

protected:
    bool initRadixStages();
    void initDigitReversal();
    bool calcRadixStages(const bool inverse);

    uint N_ = 0;
    uint nStages_ = 0;
    uint stages_[64]{ 0 };
    uint *digitReversal_ = nullptr;
    float *dstComplex_ = nullptr;
};

#endif // FFTPROTO_H

