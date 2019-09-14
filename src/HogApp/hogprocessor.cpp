#include <hogprocessor.h>

#include <QImage>
#include <QVector>

#include <fhog.hpp>

// debug
#include <iomanip>
#include <colorconversionsproto.h>

HogProcessor::HogProcessor(QObject *parent)
    : VideoProcessor(parent)
    , ocvImageGray_(new cv::Mat_<uchar>())
    , ocvImageGrayFloat_(new cv::Mat_<float>())
    , hogPiotr_(nullptr)
{
    kernelPaths_ = { "hog.cl", "colorconversions.cl" };
    timer_.start();
}

HogProcessor::~HogProcessor()
{
    release();
}

void HogProcessor::release()
{
    std::cout << "Mean processing time on " << frameIndex_ << " frames is "
        << (double)msSum_ / std::max(1, frameIndex_) << "ms\n";
    hog_.release();
    lab2rgb_.release();
    rgb2lab_.release();
    if (oclImageRgb_)
    {
        clReleaseMemObject(oclImageRgb_);
        oclImageRgb_ = NULL;
    }
    if (oclImage_)
    {
        clReleaseMemObject(oclImage_);
        oclImage_ = NULL;
    }
    hogProto_.release();
    if (hogPiotr_)
    {
        delete [] hogPiotr_;
        hogPiotr_ = nullptr;
    }
}

bool HogProcessor::setupProcessor(const VideoProcessor::CaptureSettings &settings)
{
    release();
    if (!VideoProcessor::setupProcessor(settings))
    {
        return false;
    }

    auto emitError = [this](const QString &msg)
    {
        setVideoCaptureState(CaptureState::NotInitialized);
        emit sendError(msg);
        return false;
    };

    HogSettings hogSettings;
    if (!hogSettings.init(settings.frameWidth_, settings.frameHeight_))
    {
        return emitError("Invalid image resolution passed into HogSettings");
    }
    hogProto_.initialize(hogSettings);
    {
        int bytes = hogSettings.imWidth() * hogSettings.imHeight() * sizeof(cl_float);
        oclImage_ = clCreateBuffer(oclContext_, CL_MEM_READ_ONLY, bytes, NULL, NULL);
        if (!oclImage_)
        {
            return emitError("Failed to initialize oclImage_");
        }
        oclImageRgb_ = clCreateBuffer(oclContext_, CL_MEM_READ_WRITE, bytes * 3, NULL, NULL);
        if (!oclImageRgb_)
        {
            return emitError("Failed to initialize oclImageRgb_");
        }
    }
    if (rgb2lab_.initialize(hogSettings.imWidth(), hogSettings.imHeight(),
            ColorConversion::rgb2lab, oclContext_, oclProgram_, oclImageRgb_))
    {
        return emitError("Failed to initialize Lab(rgb2lab)");
    }
    if (lab2rgb_.initialize(hogSettings.imWidth(), hogSettings.imHeight(),
            ColorConversion::lab2rgb, oclContext_, oclProgram_, rgb2lab_.converted_))
    {
        return emitError("Failed to initialize Lab(lab2rgb)");
    }
    if (hog_.initialize(hogSettings, oclContext_, oclProgram_, oclImage_) != CL_SUCCESS)
    {
        return emitError("Failed to initialize Hog");
    }

    int cellCount = hogSettings.cellCount_[0] * hogSettings.cellCount_[1];
    int length = cellCount * hogSettings.channelsPerBlock();
    hogPiotr_ = new float [length];
    std::fill(hogPiotr_, hogPiotr_ + length, 0.0f);

    emit sendHogSettings(
        hogSettings.cellCount_[0], hogSettings.cellCount_[1], hogSettings.channelsPerBlock(),
        0, hogSettings.sensitiveBinCount()
//        hogSettings.sensitiveBinCount(), hogSettings.insensitiveBinCount_
    );
    msSum_ = 0;
    return true;
}

void HogProcessor::calculateHogOcl()
{
    timer_.restart();
    cl_event imageWriteEvent = NULL;
    size_t bytes = ocvImageGrayFloat_->rows * ocvImageGrayFloat_->cols * sizeof(cl_float);
    cl_int status = clEnqueueWriteBuffer(oclQueue_, oclImage_, CL_FALSE, 0, bytes,
        ocvImageGrayFloat_->data, 0, NULL, &imageWriteEvent);
    cl_event hogEvent = NULL;
    if (status == CL_SUCCESS)
    {
        status = hog_.calculate(oclQueue_, 1, &imageWriteEvent, hogEvent);
    }
    if (imageWriteEvent)
    {
        clReleaseEvent(imageWriteEvent);
        imageWriteEvent = NULL;
    }
    bytes = hogProto_.settings_.cellCount_[0] * hogProto_.settings_.cellCount_[1] *
        hogProto_.settings_.channelsPerBlock() * sizeof(cl_float);
    cl_float *mappedDesc = NULL;
    if (status == CL_SUCCESS)
    {
        mappedDesc = (cl_float*)clEnqueueMapBuffer(oclQueue_, hog_.blockHog_.descriptor_,
            CL_TRUE, CL_MAP_READ, 0, bytes, 1, &hogEvent, NULL, &status);
    }
    quint64 ms = timer_.restart();
    msSum_ += ms;
    std::cout << frameIndex_ << ": " << ms << "ms\n";
    if (hogEvent)
    {
        clReleaseEvent(hogEvent);
        hogEvent = NULL;
    }
    if (mappedDesc)
    {
        compareDescriptorsOcl(mappedDesc);
        //compareDescriptors(mappedDesc);
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
    if (status != CL_SUCCESS)
    {
        qDebug("calculateHogOcl(...) failed");
    }
}

void HogProcessor::convertRgb2lab()
{
    timer_.restart();
    cl_event imageWriteEvent = NULL;
    size_t bytes = captureSettings_.frameHeight_ * captureSettings_.frameWidth_ * 3 * sizeof(uchar);
    cl_int status = clEnqueueWriteBuffer(oclQueue_, oclImageRgb_, CL_FALSE, 0, bytes,
        rgbFrame_, 0, NULL, &imageWriteEvent);
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
        mappedLab = (cl_uchar*)clEnqueueMapBuffer(oclQueue_, rgb2lab_.converted_,
            CL_TRUE, CL_MAP_READ, 0, bytes, 1, &lab2rgbEvent, NULL, &status);
    }
    if (lab2rgbEvent)
    {
        clReleaseEvent(lab2rgbEvent);
        lab2rgbEvent = NULL;
    }
    quint64 ms = timer_.restart();
    msSum_ += ms;
    std::cout << frameIndex_ << ": " << ms << "ms\n";
    cl_event unmapEvent = NULL;
    if (mappedLab)
    {
        int sz = captureSettings_.frameHeight_ * captureSettings_.frameWidth_;
        std::vector<uchar> gtLab(sz * 3);
        rgb2lab(rgbFrame_, sz, gtLab.data());
        compareColorConversions(mappedLab, gtLab.data());
        status = clEnqueueUnmapMemObject(oclQueue_, rgb2lab_.converted_, mappedLab,
            0, NULL, &unmapEvent);
    }
    cl_uchar *mappedRgb = NULL;
    if (status == CL_SUCCESS)
    {
        mappedRgb = (cl_uchar*)clEnqueueMapBuffer(oclQueue_, lab2rgb_.converted_,
            CL_TRUE, CL_MAP_READ, 0, bytes, 1, &unmapEvent, NULL, &status);
    }
    if (unmapEvent)
    {
        clReleaseEvent(unmapEvent);
        unmapEvent = NULL;
    }
    if (mappedRgb)
    {
        compareColorConversions(mappedRgb, rgbFrame_);
        status = clEnqueueUnmapMemObject(oclQueue_, lab2rgb_.converted_, mappedRgb,
            0, NULL, &unmapEvent);
    }
    if (unmapEvent)
    {
        clWaitForEvents(1, &unmapEvent);
        clReleaseEvent(unmapEvent);
        unmapEvent = NULL;
    }
    if (status != CL_SUCCESS)
    {
        qDebug("color conversion failed");
    }
}

void HogProcessor::calculateHogPiotr()
{
    cv::Mat ocvImage(captureSettings_.frameHeight_, captureSettings_.frameWidth_, CV_8UC3, rgbFrame_);
    cv::cvtColor(ocvImage, *ocvImageGray_, CV_RGB2GRAY);
    ocvImageGray_->convertTo(*ocvImageGrayFloat_, CV_32FC1);

    const HogSettings &hogSettings = hogProto_.settings_;
    std::vector<cv::Mat> hogPiotr = FHoG::extract(
        *ocvImageGrayFloat_, 2, hogSettings.cellSize_, hogSettings.insensitiveBinCount_,
        1, hogSettings.truncation_);
    for (int c = 0; c < hogSettings.channelsPerBlock(); ++c)
    {
        const cv::Mat &hogChannel = hogPiotr[c];
        for (int y = 0; y < hogSettings.cellCount_[1]; ++y)
        {
            for (int x = 0; x < hogSettings.cellCount_[0]; ++x)
            {
                int featureIndex = x + y * hogSettings.cellCount_[0];
                hogPiotr_[featureIndex * hogSettings.channelsPerBlock() + c] =
                    hogChannel.at<float>(y, x);
            }
        }
    }
}

void HogProcessor::compareDescriptors(const float *desc) const
{
    const HogSettings &hogSettings = hogProto_.settings_;
    int cellCount[2] = { hogSettings.cellCount_[0], hogSettings.cellCount_[1] };
    int cellCountTotal = cellCount[0] * cellCount[1];
    int channelsPerFeature = hogSettings.channelsPerBlock();

    int mismatchCount = 0;
    int mismatchCountLast4 = 0;

    for (int c = 0; c < cellCountTotal; ++c)
    {
        for (int b = 0; b < channelsPerFeature; ++b)
        {
            int cellX = c % cellCount[0];
            int cellY = c / cellCount[0];
            if (cellX < 2 || cellX >= cellCount[0] - 2 ||
                cellY < 2 || cellY >= cellCount[1] - 2)
            {
//                continue;
            }

            int channelIndex = c * channelsPerFeature + b;
            //float ours = hogProto_.blockDescriptor_[channelIndex];
            float ours = desc[channelIndex];
            float gt = hogPiotr_[channelIndex];
            float delta = gt - ours;
            if (fabsf(delta) > fmaxf(gt * 0.05f, 1e-3f))
            {
                mismatchCount++;
                mismatchCountLast4 += b >= 27;
            }
        }
    }

    int channelsTotal = cellCountTotal * channelsPerFeature;
    std::cout << "mismatched: " << mismatchCount << " from: " << channelsTotal
        << ". mismatch ratio: " << (float)mismatchCount / (float)channelsTotal
        << " with mismatched [27, 31) channels ratio: "
        << (float)mismatchCountLast4 / (float)mismatchCount << "\n";
}

void HogProcessor::compareDescriptorsOcl(const float *mappedDescriptor) const
{
    const HogSettings &hogSettings = hogProto_.settings_;
    int cellCount[2] = { hogSettings.cellCount_[0], hogSettings.cellCount_[1] };
    int binCntPerBlock = hogSettings.channelsPerBlock();

    int mismatchCount = 0;
    int nonZerosCount = 0;
    for (int y = 0; y < cellCount[1]; ++y)
    {
        for (int x = 0; x < cellCount[0]; ++x)
        {
            for (int b = 0; b < binCntPerBlock; ++b)
            {
                int c = x + y * cellCount[0];
                float ours = (float)mappedDescriptor[c * binCntPerBlock + b];// * 1e-6f;
                float gt = hogProto_.blockDescriptor_[c * binCntPerBlock + b];
                float delta = gt - ours;
                if (fabsf(delta) > fmaxf(gt, 1e-3f) * 0.01f)
                {
                    mismatchCount++;
//                    std::cout << "(" << x << ", " << y << ", " << b << ") " << delta << "\n";
                }
                nonZerosCount += fabsf(ours) > 1e-3f;
            }
        }
//        std::cout << y << ": " << mismatchCount << "\n";
    }

    int channelsTotal = cellCount[0] * cellCount[1] * binCntPerBlock;
    std::cout << "mismatched: " << mismatchCount << " from: " << channelsTotal
        << ". mismatch ratio: " << (float)mismatchCount / (float)channelsTotal
        << ". nonzeros: " << nonZerosCount << "\n";

/*    int width = ocvImageGrayFloat_->cols;
    int height = ocvImageGrayFloat_->rows;
    int mismatchCount = 0;
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            float ours = mappedDescriptor[x + y * width];
//            float gt = ocvImageGrayFloat_->at<float>(y, std::min(x + 1, width - 1)) -
//                ocvImageGrayFloat_->at<float>(y, std::max(0, x - 1));
            float gt = ocvImageGrayFloat_->at<float>(std::min(y + 1, height - 1), x) -
                ocvImageGrayFloat_->at<float>(std::max(0, y - 1), x);
            mismatchCount += ours != gt;
            if (ours != gt && x == 319)
            {
                std::cout << "(" << x << ", " << y << ")\n";
            }
        }
    }
    int total = ocvImageGrayFloat_->rows * width;
    std::cout << "mismatched: " << mismatchCount << " from: " << total
        << ". mismatch ratio: " << (float)mismatchCount / (float)total << "\n";*/
}

void HogProcessor::compareColorConversions(const cl_uchar *ours, const uchar *gt) const
{
    int sz = captureSettings_.frameHeight_ * captureSettings_.frameWidth_;
    int cnt[3] = { 0, 0, 0 };
    for (int pix = 0; pix < sz; ++pix)
    {
        for (int c = 0; c < 3; ++c)
        {
            int i = pix * 3 + c;
            cnt[c] += std::abs((int)ours[i] - (int)gt[i]) > 3;
        }
    }
    std::cout << "mismatch: [" << cnt[0] << ", " << cnt[1] << ", " << cnt[2] << "] ";
    std::cout << "percent: [" << (float)cnt[0] / sz << ", " << (float)cnt[1] / sz
        << ", " << (float)cnt[2] / sz << "]\n";
}

bool HogProcessor::processFrame()
{
    if (!captureFrame())
    {
        qDebug("Failed to capture %d-th frame", frameIndex_);
        return false;
    }

    calculateHogPiotr();
    hogProto_.calculate((float*)ocvImageGrayFloat_->data);
    //calculateHogOcl();
    convertRgb2lab();
    compareDescriptors(hogProto_.blockDescriptor_);

    QImage qimage(rgbFrame_, captureSettings_.frameWidth_, captureSettings_.frameHeight_,
        captureSettings_.frameWidth_ * 3, QImage::Format_RGB888);
    {
        QImage lab(qimage.width(), qimage.height(), QImage::Format_RGB888);
        int sz = qimage.width() * qimage.height();
        rgb2lab(qimage.bits(), sz, lab.bits());
        lab2rgb(lab.bits(), sz, qimage.bits());
    }
    emit sendFrame(qimage.copy());

    const HogSettings &s = hogProto_.settings_;
    QVector<float> container(s.cellCount_[0] * s.cellCount_[1] * s.channelsPerBlock(), 0.0f);
    const float *desc = hogProto_.blockDescriptor_; // hogPiotr_;
    qCopy(desc, desc + container.size(), container.begin());
    emit sendHog(container);
    return true;
}

