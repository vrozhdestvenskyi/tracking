#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <fftproto.h>
#include <testhelpers.h>

std::vector<float> generateRandomData(const uint N)
{
    srand(4);
    std::vector<float> x;
    x.reserve(N);
    for (uint i = 0; i < N; ++i)
    {
        const float r = static_cast<float>(rand()) / RAND_MAX;
        x.push_back((r - 0.5f) * 200.0f);
    }
    return x;
}

std::vector<float> generateRandomData(const std::vector<uint> &stages)
{
    uint N = 1U;
    for (const uint Ny : stages)
    {
        N *= Ny;
    }
    return generateRandomData(N);
}

TEST(FftTest, ocvForwardInverse)
{
    const std::vector<float> src{ 12.345f, -1.0f, 42.0f, 0.0f, 0.0f, -0.05f, 10.0f, 3.14159265f };
    const float eps = 1e-6f;

    cv::Mat transformed;
    cv::dft(src, transformed, cv::DFT_COMPLEX_OUTPUT);
    cv::Mat backTransformed;
    cv::dft(transformed, backTransformed, cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_COMPLEX_OUTPUT);

    ASSERT_EQ(backTransformed.cols, src.size());
    ASSERT_EQ(backTransformed.type(), CV_32FC2);
    for (int col = 0; col < backTransformed.cols; ++col)
    {
        const auto &complex = backTransformed.at<cv::Vec2f>(0, col);
        ASSERT_LE(fabsf(complex[0] - src[col]), eps);
        ASSERT_LE(fabsf(complex[1]), eps);
    }
}

TEST(FftTest, protoAgainstOpenCV)
{
    const std::vector<float> src{ 12.345f, -1.0f, 42.0f, 0.0f, 0.0f, -0.05f, 10.0f, 3.14159265f };
    const float eps = 1e-5f;

    cv::Mat ocv;
    cv::dft(src, ocv, cv::DFT_COMPLEX_OUTPUT);

    FftProto ours;
    ASSERT_TRUE(ours.init(src.size()));
    ASSERT_TRUE(ours.calcForward(src.data()));

    for (size_t i = 0; i < src.size(); ++i)
    {
        for (uint dim = 0; dim < 2; ++dim)
        {
            ASSERT_LE(fabsf(ours.result()[2 * i + dim] - ocv.at<cv::Vec2f>(0, i)[dim]), eps);
        }
    }
}

TEST(FftTest, protoForwardInverse)
{
    const std::vector<uint> stages{ 2, 3, 4, 5, 6, 7, 8 };
    const std::vector<float> src = generateRandomData(stages);
    const float eps = 1e-4f;

    FftProto ours;
    ASSERT_TRUE(ours.init(src.size(), &stages));
    ASSERT_TRUE(ours.calcForward(src.data()));
    std::vector<float> transformed(src.size() * 2, 0.0f);
    std::copy(ours.result(), ours.result() + transformed.size(), transformed.data());
    ASSERT_TRUE(ours.calc(transformed.data(), true));

    for (size_t i = 0; i < src.size(); ++i)
    {
        ASSERT_LE(fabsf(ours.result()[2 * i] - src[i]), eps);
        ASSERT_LE(fabsf(ours.result()[2 * i + 1]), eps);
    }
}

TEST(FftTest, proto2dForwardInverse)
{
    const uint width = 64U;
    const uint height = 48U;
    const std::vector<float> src = generateRandomData(width * height);
    const float eps = 1e-4f;

    Fft2dProto ours;
    ASSERT_TRUE(ours.init(width, height));
    ASSERT_TRUE(ours.calcForward(src.data()));
    std::vector<float> transformed(src.size() * 2, 0.0f);
    std::copy(ours.result(), ours.result() + transformed.size(), transformed.data());
    ASSERT_TRUE(ours.calc(transformed.data(), true));

    for (size_t i = 0; i < src.size(); ++i)
    {
        ASSERT_LE(fabsf(ours.result()[2 * i] - src[i]), eps);
        ASSERT_LE(fabsf(ours.result()[2 * i + 1]), eps);
    }
}

