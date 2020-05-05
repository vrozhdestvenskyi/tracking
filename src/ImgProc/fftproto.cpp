#include <fftproto.h>
#include <algorithm>

FftProto::FftProto()
{
    std::fill(std::begin(stages_), std::end(stages_), 0U);
}

FftProto::~FftProto()
{
    release();
}

void FftProto::release()
{
    if (dstComplex_)
    {
        delete [] dstComplex_;
        dstComplex_ = nullptr;
    }
    if (digitReversal_)
    {
        delete [] digitReversal_;
        digitReversal_ = nullptr;
    }
}

bool FftProto::initRadixStages()
{
    uint N = N_;
    nStages_ = 0U;
    for (uint Ny : {3U, 2U})
    {
        while (N % Ny == 0U)
        {
            stages_[nStages_++] = Ny;
            N /= Ny;
        }
    }
    return N == 1U;
}

void FftProto::initDigitReversal()
{
    for (uint n = 0; n < N_; ++n)
    {
        uint k = n;
        uint Nx = stages_[0];
        for (uint stageId = 1; stageId < nStages_; ++stageId)
        {
            const uint Ny = stages_[stageId];
            const uint Ni = Ny * Nx;
            k = (k * Ny) % Ni + (k / Nx) % Ny + Ni * (k / Ni);
            Nx *= Ny;
        }
        digitReversal_[n] = k;
    }
}

bool FftProto::init(const uint N)
{
    release();
    N_ = N;
    if (!initRadixStages())
    {
        return false;
    }
    digitReversal_ = new uint [N];
    std::fill(digitReversal_, digitReversal_ + N, 0);
    initDigitReversal();
    dstComplex_ = new float [2 * N];
    std::fill(dstComplex_, dstComplex_ + 2 * N, 0.0f);
    return true;
}

// The idea is taken from the ARM's blog-post "Speeding-up Fast Fourier Transform
// Mixed-Radix on Mali GPU with OpenCL" (the link is splitted into 3 lines only in order
// not to destroy the code redactor, so please remove the endlines and comment symbols):
// https://community.arm.com/developer/tools-software/graphics/b/blog/posts/
// speeding-up-fast-fourier-transform-mixed-radix-on-mobile-arm-mali-gpu-by-
// means-of-opencl---part-1
// All the variables names are taken from the main equation (just compile this at some
// Latex editor):
// $X(k_y + k_x \cdot N_y) = \sum \limits_{n_x = 0}^{N_x - 1}
// e^{-\frac{2 \cdot \pi \cdot i \cdot k_y \cdot n_x}{N_x \cdot N_y}} \cdot
// \left \{ \sum \limits_{n_y = 0}^{N_y - 1} x(n_x + n_y \cdot N_x) \cdot
// e^{-\frac{2 \cdot \pi \cdot i \cdot k_y \cdot n_y}{N_y}} \right \} \cdot
// e^{-\frac{2 \cdot \pi \cdot i \cdot k_x \cdot n_x}{N_x}}$
bool FftProto::calcRadixStages(const bool inverse)
{
    float x[3][2]{ { 0.0f } };
    uint Nx = 1U;
    for (uint stageId = 0; stageId < nStages_; ++stageId)
    {
        const uint Ny = stages_[stageId];
        const uint Ni = Nx * Ny;
        for (uint kx = 0; kx < N_ / Ny; ++kx)
        {
            // Load
            const uint nx = kx % Nx;
            const uint n = nx + (kx / Nx) * Ni; // actually this is not exactly n but almost n
            for (uint ky = 0; ky < Ny; ++ky)
            {
                for (uint i = 0; i < 2; ++i)
                {
                    x[ky][i] = dstComplex_[2 * (n + ky * Nx) + i];
                }
            }
            // Twiddle factors multiplication
            for (uint ky = 0; ky < Ny; ++ky)
            {
                const float phi = nx * ky * static_cast<float>(-2 * M_PI) / Ni;
                const float w[2]{ cosf(phi), sinf(phi) * (inverse ? -1.0f : 1.0f) };
                const float tmp = w[0] * x[ky][0] - w[1] * x[ky][1];
                x[ky][1] = w[0] * x[ky][1] + w[1] * x[ky][0];
                x[ky][0] = tmp;
            }
            // Radix computation
            switch (Ny)
            {
            case 3:
            {
                const float SQRT3DIV2 = 0.86602540378443f * (inverse ? -1.0f : 1.0f);
                const float v0[2]{ x[1][0] + x[2][0], x[1][1] + x[2][1] };
                const float v1[2]{ x[1][0] - x[2][0], x[1][1] - x[2][1] };
                x[1][0] = x[0][0] - 0.5f * v0[0] + v1[1] * SQRT3DIV2;
                x[1][1] = x[0][1] - 0.5f * v0[1] - v1[0] * SQRT3DIV2;
                x[2][0] = x[0][0] - 0.5f * v0[0] - v1[1] * SQRT3DIV2;
                x[2][1] = x[0][1] - 0.5f * v0[1] + v1[0] * SQRT3DIV2;
                x[0][0] = x[0][0] + v0[0];
                x[0][1] = x[0][1] + v0[1];
                break;
            }
            case 2:
                for (uint i = 0; i < 2; ++i)
                {
                    const float v = x[0][i];
                    x[0][i] = v + x[1][i];
                    x[1][i] = v - x[1][i];
                }
                break;
            default:
                return false;
            }
            // Store
            for (uint ky = 0; ky < Ny; ++ky)
            {
                for (uint i = 0; i < 2; ++i)
                {
                    dstComplex_[2 * (n + ky * Nx) + i] = x[ky][i];
                }
            }
        }
        Nx *= Ny;
    }
    return true;
}

bool FftProto::calcForward(const float *srcReal)
{
    for (uint n = 0; n < N_; ++n)
    {
        const uint k = digitReversal_[n];
        dstComplex_[2 * n] = srcReal[k];
        dstComplex_[2 * n + 1] = 0.0f;
    }
    return calcRadixStages(false);
}

bool FftProto::calc(const float *srcComplex, const bool inverse)
{
    for (uint n = 0; n < N_; ++n)
    {
        const uint k = digitReversal_[n];
        dstComplex_[2 * n] = srcComplex[2 * k];
        dstComplex_[2 * n + 1] = srcComplex[2 * k + 1];
    }
    if (!calcRadixStages(inverse))
    {
        return false;
    }
    if (inverse)
    {
        const float scale = 1.0f / N_;
        for (uint i = 0; i < 2 * N_; ++i)
        {
            dstComplex_[i] *= scale;
        }
    }
    return true;
}

const float * FftProto::result() const
{
    return dstComplex_;
}

