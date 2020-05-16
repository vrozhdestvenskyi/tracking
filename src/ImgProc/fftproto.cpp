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
    for (uint Ny : {8U, 7U, 6U, 5U, 4U, 3U, 2U})
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
    float x[8][2]{ {0.0f} };
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
            case 8U:
            {
                const float SQRT2DIV2 = 0.70710678118654f;
                float v0[2], v1[2], v2[2], v3[2], v4[2], v5[2], v6[2], v7[2];
                for (uint i = 0; i < 2; ++i)
                {
                    v0[i] = x[0][i] + x[4][i];
                    v1[i] = x[0][i] - x[4][i];
                    v2[i] = x[1][i] + x[3][i];
                    v3[i] = x[1][i] - x[3][i];
                    v4[i] = x[2][i] + x[6][i];
                    v5[i] = x[2][i] - x[6][i];
                    v6[i] = x[5][i] + x[7][i];
                    v7[i] = x[5][i] - x[7][i];
                }
                for (uint i = 0; i < 2; ++i)
                {
                    x[0][i] = v0[i] + v2[i] + v4[i] + v6[i];
                    x[4][i] = v0[i] - v2[i] + v4[i] - v6[i];
                }
                x[2][0] = v0[0] - v4[0] + (v3[1] + v7[1]) * (inverse ? -1.0f : 1.0f);
                x[2][1] = v0[1] - v4[1] - (v3[0] + v7[0]) * (inverse ? -1.0f : 1.0f);
                x[6][0] = v0[0] - v4[0] - (v3[1] + v7[1]) * (inverse ? -1.0f : 1.0f);
                x[6][1] = v0[1] - v4[1] + (v3[0] + v7[0]) * (inverse ? -1.0f : 1.0f);
                for (uint i = 0; i < 2; ++i)
                {
                    v2[i] *= SQRT2DIV2;
                    v3[i] *= SQRT2DIV2;
                    v6[i] *= SQRT2DIV2;
                    v7[i] *= SQRT2DIV2;
                }
                x[1][0] = v1[0] + v3[0] - v7[0] + (v2[1] + v5[1] - v6[1]) * (inverse ? -1.0f : 1.0f);
                x[1][1] = v1[1] + v3[1] - v7[1] - (v2[0] + v5[0] - v6[0]) * (inverse ? -1.0f : 1.0f);
                x[3][0] = v1[0] - v3[0] + v7[0] + (v2[1] - v5[1] - v6[1]) * (inverse ? -1.0f : 1.0f);
                x[3][1] = v1[1] - v3[1] + v7[1] - (v2[0] - v5[0] - v6[0]) * (inverse ? -1.0f : 1.0f);
                x[5][0] = v1[0] - v3[0] + v7[0] - (v2[1] - v5[1] - v6[1]) * (inverse ? -1.0f : 1.0f);
                x[5][1] = v1[1] - v3[1] + v7[1] + (v2[0] - v5[0] - v6[0]) * (inverse ? -1.0f : 1.0f);
                x[7][0] = v1[0] + v3[0] - v7[0] - (v2[1] + v5[1] - v6[1]) * (inverse ? -1.0f : 1.0f);
                x[7][1] = v1[1] + v3[1] - v7[1] + (v2[0] + v5[0] - v6[0]) * (inverse ? -1.0f : 1.0f);
                break;
            }
            case 7U:
            {
                const float W7A = 0.62348980185873f;
                const float W7B = 0.78183148246802f * (inverse ? -1.0f : 1.0f);
                const float W7C = 0.22252093395631f;
                const float W7D = 0.97492791218182f * (inverse ? -1.0f : 1.0f);
                const float W7E = 0.90096886790241f;
                const float W7F = 0.43388373911755f * (inverse ? -1.0f : 1.0f);
                float v0[2], v1[2], v2[2], v3[2], v4[2], v5[2], v6[2];
                for (uint i = 0; i < 2; ++i)
                {
                    v0[i] = x[0][i];
                    v1[i] = W7A * (x[1][i] + x[6][i]) - W7C * (x[2][i] + x[5][i]) - W7E * (x[3][i] + x[4][i]);
                    v2[i] = W7C * (x[1][i] + x[6][i]) + W7E * (x[2][i] + x[5][i]) - W7A * (x[3][i] + x[4][i]);
                    v3[i] = W7E * (x[1][i] + x[6][i]) - W7A * (x[2][i] + x[5][i]) + W7C * (x[3][i] + x[4][i]);
                    v4[i] = W7B * (x[1][i] - x[6][i]) + W7D * (x[2][i] - x[5][i]) + W7F * (x[3][i] - x[4][i]);
                    v5[i] = W7D * (x[1][i] - x[6][i]) - W7F * (x[2][i] - x[5][i]) - W7B * (x[3][i] - x[4][i]);
                    v6[i] = W7F * (x[1][i] - x[6][i]) - W7B * (x[2][i] - x[5][i]) + W7D * (x[3][i] - x[4][i]);
                }
                for (uint i = 0; i < 2; ++i)
                {
                    x[0][i] = v0[i] + x[1][i] + x[2][i] + x[3][i] + x[4][i] + x[5][i] + x[6][i];
                }
                x[1][0] = v0[0] + v1[0] + v4[1];
                x[1][1] = v0[1] + v1[1] - v4[0];
                x[2][0] = v0[0] - v2[0] + v5[1];
                x[2][1] = v0[1] - v2[1] - v5[0];
                x[3][0] = v0[0] - v3[0] + v6[1];
                x[3][1] = v0[1] - v3[1] - v6[0];
                x[4][0] = v0[0] - v3[0] - v6[1];
                x[4][1] = v0[1] - v3[1] + v6[0];
                x[5][0] = v0[0] - v2[0] - v5[1];
                x[5][1] = v0[1] - v2[1] + v5[0];
                x[6][0] = v0[0] + v1[0] - v4[1];
                x[6][1] = v0[1] + v1[1] + v4[0];
                break;
            }
            case 6U:
            {
                const float SQRT3DIV2 = 0.86602540378443f * (inverse ? -1.0f : 1.0f);
                float v0[2], v1[2], v2[2], v3[3], v4[4], v5[5];
                for (uint i = 0; i < 2; ++i)
                {
                    v0[i] = x[0][i] + x[3][i];
                    v1[i] = x[0][i] - x[3][i];
                    v2[i] = x[1][i] + x[2][i];
                    v3[i] = x[1][i] - x[2][i];
                    v4[i] = x[4][i] + x[5][i];
                    v5[i] = x[4][i] - x[5][i];
                }
                for (uint i = 0; i < 2; ++i)
                {
                    x[0][i] = v0[i] + v2[i] + v4[i];
                    x[3][i] = v1[i] - v3[i] + v5[i];
                }
                x[1][0] = v1[0] + (v3[0] - v5[0]) * 0.5f - (v4[1] - v2[1]) * SQRT3DIV2;
                x[1][1] = v1[1] + (v3[1] - v5[1]) * 0.5f + (v4[0] - v2[0]) * SQRT3DIV2;
                x[2][0] = v0[0] - (v2[0] + v4[0]) * 0.5f + (v3[1] + v5[1]) * SQRT3DIV2;
                x[2][1] = v0[1] - (v2[1] + v4[1]) * 0.5f - (v3[0] + v5[0]) * SQRT3DIV2;
                x[4][0] = v0[0] - (v2[0] + v4[0]) * 0.5f - (v3[1] + v5[1]) * SQRT3DIV2;
                x[4][1] = v0[1] - (v2[1] + v4[1]) * 0.5f + (v3[0] + v5[0]) * SQRT3DIV2;
                x[5][0] = v1[0] + (v3[0] - v5[0]) * 0.5f - (v2[1] - v4[1]) * SQRT3DIV2;
                x[5][1] = v1[1] + (v3[1] - v5[1]) * 0.5f + (v2[0] - v4[0]) * SQRT3DIV2;
                break;
            }
            case 5U:
            {
                const float W5A = 0.30901699437494f;
                const float W5B = 0.95105651629515f * (inverse ? -1.0f : 1.0f);
                const float W5C = 0.80901699437494f;
                const float W5D = 0.58778525229247f * (inverse ? -1.0f : 1.0f);
                float v0[2], v1[2], v2[2], v3[2], v4[2];
                for (uint i = 0; i < 2; ++i)
                {
                    v0[i] = x[0][i];
                    v1[i] = W5A * (x[1][i] + x[4][i]) - W5C * (x[2][i] + x[3][i]);
                    v2[i] = W5C * (x[1][i] + x[4][i]) - W5A * (x[2][i] + x[3][i]);
                    v3[i] = W5D * (x[1][i] - x[4][i]) - W5B * (x[2][i] - x[3][i]);
                    v4[i] = W5B * (x[1][i] - x[4][i]) + W5D * (x[2][i] - x[3][i]);
                    x[0][i] = v0[i] + x[1][i] + x[2][i] + x[3][i] + x[4][i];
                }
                x[1][0] = v0[0] + v1[0] + v4[1];
                x[1][1] = v0[1] + v1[1] - v4[0];
                x[2][0] = v0[0] - v2[0] + v3[1];
                x[2][1] = v0[1] - v2[1] - v3[0];
                x[3][0] = v0[0] - v2[0] - v3[1];
                x[3][1] = v0[1] - v2[1] + v3[0];
                x[4][0] = v0[0] + v1[0] - v4[1];
                x[4][1] = v0[1] + v1[1] + v4[0];
                break;
            }
            case 4U:
            {
                const float v3[2]{
                    (x[1][1] - x[3][1]) * (inverse ? -1.0f : 1.0f),
                    (x[3][0] - x[1][0]) * (inverse ? -1.0f : 1.0f) };
                for (uint i = 0; i < 2; ++i)
                {
                    const float v0 = x[0][i] + x[2][i];
                    const float v1 = x[1][i] + x[3][i];
                    const float v2 = x[0][i] - x[2][i];
                    x[0][i] = v0 + v1;
                    x[2][i] = v0 - v1;
                    x[1][i] = v2 + v3[i];
                    x[3][i] = v2 - v3[i];
                }
                break;
            }
            case 3U:
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
            case 2U:
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

