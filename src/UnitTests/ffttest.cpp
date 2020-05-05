#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <fftproto.h>
#include <testhelpers.h>

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
        for (int dim = 0; dim < 2; ++dim)
        {
            ASSERT_LE(fabsf(ours.result()[2 * i + dim] - ocv.at<cv::Vec2f>(0, i)[dim]), eps);
        }
    }
}

TEST(FftTest, protoForwardInverse)
{
    const std::vector<float> src{ 12.345f, -1.0f, 42.0f, 0.0f, 0.0f, -0.05f, 10.0f, 3.14159265f };
    const float eps = 1e-6f;

    FftProto ours;
    ASSERT_TRUE(ours.init(src.size()));
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

