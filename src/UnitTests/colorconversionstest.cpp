#include <iostream>
#include <QImage>
#include <opencv2/opencv.hpp>
#include <gtest/gtest.h>
#include <colorconversionsproto.h>
#include <colorconversions.h>
#include <oclprocessor.h>
#include <testhelpers.h>

class TestProcessor : public OclProcessor
{
public:
    TestProcessor()
    {
        kernelPaths_ = { "colorconversions.cl" };
    }

    ~TestProcessor()
    {
        release();
    }

    bool setup(int width, int height)
    {
        release();
        if (OclProcessor::initialize() != CL_SUCCESS)
        {
            return false;
        }
        width_ = width;
        height_ = height;
        oclImage_ = clCreateBuffer(oclContext_, CL_MEM_READ_ONLY, imSizeInBytes(), NULL, NULL);
        if (!oclImage_)
        {
            return false;
        }
        if (rgb2lab_.initialize(width, height, ColorConversion::rgb2lab,
                oclContext_, oclProgram_, oclImage_))
        {
            return false;
        }
        if (lab2rgb_.initialize(width, height, ColorConversion::lab2rgb,
                oclContext_, oclProgram_, rgb2lab_.converted_))
        {
            return false;
        }
        return true;
    }

    bool processFrame(const uchar *srcRgb, uchar *dstLab, uchar *dstRgb)
    {
        cl_event imageWriteEvent = NULL;
        cl_int status = clEnqueueWriteBuffer(oclQueue_, oclImage_, CL_FALSE, 0,
            imSizeInBytes(), srcRgb, 0, NULL, &imageWriteEvent);
        cl_event rgb2labEvent = NULL;
        if (status == CL_SUCCESS)
        {
            status = rgb2lab_.calculate(oclQueue_, 1, &imageWriteEvent, rgb2labEvent);
        }
        if (imageWriteEvent)
        {
            clReleaseEvent(imageWriteEvent);
            imageWriteEvent = NULL;
        }
        cl_event lab2rgbEvent = NULL;
        if (status == CL_SUCCESS)
        {
            status = lab2rgb_.calculate(oclQueue_, 1, &rgb2labEvent, lab2rgbEvent);
        }
        if (rgb2labEvent)
        {
            clReleaseEvent(rgb2labEvent);
            rgb2labEvent = NULL;
        }
        cl_uchar *mappedLab = NULL;
        if (status == CL_SUCCESS)
        {
            mappedLab = (cl_uchar*)clEnqueueMapBuffer(oclQueue_, rgb2lab_.converted_, CL_TRUE,
                CL_MAP_READ, 0, imSizeInBytes(), 1, &lab2rgbEvent, NULL, &status);
        }
        if (lab2rgbEvent)
        {
            clReleaseEvent(lab2rgbEvent);
            lab2rgbEvent = NULL;
        }
        cl_event unmapEvent = NULL;
        if (mappedLab)
        {
            std::copy(mappedLab, mappedLab + imSizeInBytes(), dstLab);
            status = clEnqueueUnmapMemObject(oclQueue_, rgb2lab_.converted_, mappedLab,
                0, NULL, &unmapEvent);
        }
        cl_uchar *mappedRgb = NULL;
        if (status == CL_SUCCESS)
        {
            mappedRgb = (cl_uchar*)clEnqueueMapBuffer(oclQueue_, lab2rgb_.converted_, CL_TRUE,
                CL_MAP_READ, 0, imSizeInBytes(), 1, &unmapEvent, NULL, &status);
        }
        if (unmapEvent)
        {
            clReleaseEvent(unmapEvent);
            unmapEvent = NULL;
        }
        if (mappedRgb)
        {
            std::copy(mappedRgb, mappedRgb + imSizeInBytes(), dstRgb);
            status = clEnqueueUnmapMemObject(oclQueue_, lab2rgb_.converted_, mappedRgb,
                0, NULL, &unmapEvent);
        }
        if (unmapEvent)
        {
            clWaitForEvents(1, &unmapEvent);
            clReleaseEvent(unmapEvent);
            unmapEvent = NULL;
        }
        return status == CL_SUCCESS;
    }

protected:
    void release()
    {
        lab2rgb_.release();
        rgb2lab_.release();
        if (oclImage_)
        {
            clReleaseMemObject(oclImage_);
            oclImage_ = NULL;
        }
    }

    int imSizeInBytes() const
    {
        return width_ * height_ * 3 * sizeof(uchar);
    }

    int width_ = 0;
    int height_ = 0;
    cl_mem oclImage_ = NULL;
    Lab rgb2lab_;
    Lab lab2rgb_;
};

void verifyEquality(const uchar *src, const uchar *dst, int width, int height)
{
    const int pixDiffThr = 3;
    const float mismatchRatioThr = 1e-4f;
    const int sz = width * height;
    int cnt[3] = { 0, 0, 0 };
    for (int pix = 0; pix < sz; ++pix)
    {
        for (int c = 0; c < 3; ++c)
        {
            const int i = pix * 3 + c;
            cnt[c] += std::abs((int)src[i] - (int)dst[i]) > pixDiffThr;
        }
    }
    for (int c = 0; c < 3; ++c)
    {
        std::cout << cnt[c] << "(" << (float)cnt[c] / sz << ") ";
        ASSERT_LT((float)cnt[c] / sz, mismatchRatioThr);
    }
    std::cout << "\n";
}

TEST(ColorConversionsTest, ProtoAgainstOpenCV)
{
    const QImage srcRgb = loadTestImage();
    const int sz = srcRgb.width() * srcRgb.height();

    const cv::Mat ocvRgb(srcRgb.height(), srcRgb.width(), CV_8UC3, (void*)srcRgb.bits());
    cv::Mat ocvLab(srcRgb.height(), srcRgb.width(), CV_8UC3);
    cv::cvtColor(ocvRgb, ocvLab, CV_RGB2Lab);

    QImage oursLab(srcRgb.width(), srcRgb.height(), srcRgb.format());
    QImage oursRgb(srcRgb.width(), srcRgb.height(), srcRgb.format());
    rgb2lab(srcRgb.bits(), sz, oursLab.bits());
    lab2rgb(oursLab.bits(), sz, oursRgb.bits());

    verifyEquality(oursLab.bits(), ocvLab.data, srcRgb.width(), srcRgb.height());
    verifyEquality(oursRgb.bits(), srcRgb.bits(), srcRgb.width(), srcRgb.height());
}

TEST(ColorConversionsTest, OclAgainstProto)
{
    TestImageSettings s;
    TestProcessor p;
    ASSERT_TRUE(p.setup(s.width_, s.height_));

    QImage srcRgb = loadTestImage();
    QImage protoLab(srcRgb.width(), srcRgb.height(), srcRgb.format());
    rgb2lab(srcRgb.bits(), srcRgb.width() * srcRgb.height(), protoLab.bits());

    QImage oursLab(srcRgb.width(), srcRgb.height(), srcRgb.format());
    QImage oursRgb(srcRgb.width(), srcRgb.height(), srcRgb.format());
    ASSERT_TRUE(p.processFrame(srcRgb.bits(), oursLab.bits(), oursRgb.bits()));

    verifyEquality(oursLab.bits(), protoLab.bits(), srcRgb.width(), srcRgb.height());
    verifyEquality(oursRgb.bits(), srcRgb.bits(), srcRgb.width(), srcRgb.height());
}

