#include <opencv2/opencv.hpp>
#include <gtest/gtest.h>
#include <fhog.hpp>
#include <hog.h>
#include <oclprocessor.h>
#include <testhelpers.h>

class TestProcessor : public OclProcessor
{
public:
    TestProcessor()
    {
        kernelPaths_ = { "hog.cl" };
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
        if (!sett_.init(width, height))
        {
            return false;
        }
        oclImGrayFloat_ = clCreateBuffer(oclContext_, CL_MEM_READ_ONLY, imSzInBytes(), NULL, NULL);
        if (!oclImGrayFloat_)
        {
            return false;
        }
        return hog_.initialize(sett_, oclContext_, oclProgram_, oclImGrayFloat_) == CL_SUCCESS;
    }

    bool processFrame(const float *im, float *desc)
    {
        cl_event imWriteEvent = NULL;
        cl_int status = clEnqueueWriteBuffer(oclQueue_, oclImGrayFloat_, CL_FALSE, 0,
            imSzInBytes(), im, 0, NULL, &imWriteEvent);
        cl_event hogEvent = NULL;
        if (status == CL_SUCCESS)
        {
            status = hog_.calculate(oclQueue_, 1, &imWriteEvent, hogEvent);
        }
        if (imWriteEvent)
        {
            clReleaseEvent(imWriteEvent);
            imWriteEvent = NULL;
        }
        cl_float *mappedDesc = NULL;
        if (status == CL_SUCCESS)
        {
            mappedDesc = (cl_float*)clEnqueueMapBuffer(oclQueue_, hog_.blockHog_.descriptor_,
                CL_TRUE, CL_MAP_READ, 0, descSzInBytes(), 1, &hogEvent, NULL, &status);
        }
        if (hogEvent)
        {
            clReleaseEvent(hogEvent);
            hogEvent = NULL;
        }
        if (mappedDesc)
        {
            std::copy(mappedDesc, mappedDesc + descLen(), desc);
        }
        cl_event unmapEvent = NULL;
        if (mappedDesc)
        {
            status = clEnqueueUnmapMemObject(oclQueue_, hog_.blockHog_.descriptor_, mappedDesc,
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
    int imSzInBytes() const
    {
        return sett_.imWidth() * sett_.imHeight() * sizeof(cl_float);
    }

    int descLen() const
    {
        return sett_.cellCount_[0] * sett_.cellCount_[1] * sett_.channelsPerBlock();
    }

    int descSzInBytes() const
    {
        return descLen() * sizeof(cl_float);
    }

    void release()
    {
        hog_.release();
        if (oclImGrayFloat_)
        {
            clReleaseMemObject(oclImGrayFloat_);
            oclImGrayFloat_ = NULL;
        }
    }

    HogSettings sett_;
    cl_mem oclImGrayFloat_ = nullptr;
    std::vector<float> desc;
    Hog hog_;
};

class HogTest : public ::testing::Test
{
public:
    static void SetUpTestSuite()
    {
        const QImage srcRgb = loadTestImage();
        cv::Mat ocvRgb(srcRgb.height(), srcRgb.width(), CV_8UC3, (void*)srcRgb.bits());
        cv::Mat ocvGray;
        cv::cvtColor(ocvRgb, ocvGray, CV_RGB2GRAY);
        ocvGray.convertTo(ocvImGrayFloat_, CV_32FC1);
        ASSERT_TRUE(sett_.init(ocvImGrayFloat_.cols, ocvImGrayFloat_.rows));
    }

protected:
    static int descLen()
    {
        return sett_.cellCount_[0] * sett_.cellCount_[1] * sett_.channelsPerBlock();
    }

    std::vector<float> calcPiotr() const
    {
        std::vector<cv::Mat> descPiotr = FHoG::extract(ocvImGrayFloat_, 2, sett_.cellSize_,
            sett_.insensitiveBinCount_, 1, sett_.truncation_);
        std::vector<float> desc(descLen(), 0.0f);
        for (int c = 0; c < sett_.channelsPerBlock(); ++c)
        {
            const cv::Mat &channel = descPiotr[c];
            for (int y = 0; y < sett_.cellCount_[1]; ++y)
            {
                for (int x = 0; x < sett_.cellCount_[0]; ++x)
                {
                    int i = x + y * sett_.cellCount_[0];
                    desc[i * sett_.channelsPerBlock() + c] = channel.at<float>(y, x);
                }
            }
        }
        return desc;
    }

    void compareDescriptors(const float *src, const float *dst, float thrRelativeMismatchCnt) const
    {
        int mismatch = 0;
        for (int i = 0; i < descLen(); ++i)
        {
            mismatch =+ fabsf(src[i] - dst[i]) > fmaxf(dst[i] * 0.05f, 1e-3f);
        }
        ASSERT_GT(descLen() * thrRelativeMismatchCnt, mismatch);
        std::cout << "mismatched " << mismatch << "(" << (float)mismatch / descLen() << ")\n";
    }

    static HogSettings sett_;
    static cv::Mat ocvImGrayFloat_;
};

cv::Mat HogTest::ocvImGrayFloat_;
HogSettings HogTest::sett_;

TEST_F(HogTest, protoAgainstPiotr)
{
    HogProto proto;
    proto.initialize(sett_);
    proto.calculate((float*)ocvImGrayFloat_.data);
    std::vector<float> piotr = calcPiotr();
    compareDescriptors(proto.blockDescriptor_, piotr.data(), 1e-5f);
}

TEST_F(HogTest, oclAgainstProto)
{
    HogProto proto;
    proto.initialize(sett_);
    proto.calculate((float*)ocvImGrayFloat_.data);
    TestProcessor ocl;
    ASSERT_TRUE(ocl.setup(sett_.imWidth(), sett_.imHeight()));
    std::vector<float> oclDesc(descLen(), 0.0f);
    ASSERT_TRUE(ocl.processFrame((float*)ocvImGrayFloat_.data, oclDesc.data()));
    compareDescriptors(oclDesc.data(), proto.blockDescriptor_, 1e-5f);
}

