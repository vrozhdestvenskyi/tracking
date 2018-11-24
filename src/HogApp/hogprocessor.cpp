#include <hogprocessor.h>

#include <QImage>
#include <QVector>

#include <fhog.hpp>

// debug
#include <iomanip>

HogProcessor::HogProcessor(QObject *parent)
    : VideoProcessor(parent)
    , ocvImageGray_(new cv::Mat_<uchar>())
    , ocvImageGrayFloat_(new cv::Mat_<float>())
    , hogPiotr_(nullptr)
{
    kernelPaths_ = { "hog.cl" };
}

HogProcessor::~HogProcessor()
{
    release();
}

void HogProcessor::release()
{
    hog_.release();
    hogProto_.release();
    if (hogPiotr_)
    {
        delete [] hogPiotr_;
        hogPiotr_ = nullptr;
    }
}

void HogProcessor::setupProcessor(const VideoProcessor::CaptureSettings &settings)
{
    release();
    VideoProcessor::setupProcessor(settings);

    auto emitError = [this](const QString &msg)
    {
        setVideoCaptureState(CaptureState::NotInitialized);
        emit sendError(msg);
    };

    HogSettings hogSettings(settings.frameWidth_, settings.frameHeight_);
    hogProto_.initialize(hogSettings);
    if (hog_.initialize(hogSettings, oclContext_, oclProgram_) != CL_SUCCESS)
    {
        emitError("Failed to initialize Hog");
        return;
    }

    int cellCount = hogSettings.cellCount_[0] * hogSettings.cellCount_[1];
    int length = cellCount * hogSettings.channelsPerFeature();
    hogPiotr_ = new float [length];
    std::fill(hogPiotr_, hogPiotr_ + length, 0.0f);

    emit sendHogSettings(
        hogSettings.cellCount_[0], hogSettings.cellCount_[1], hogSettings.channelsPerFeature(),
        0, hogSettings.sensitiveBinCount()
//        hogSettings.sensitiveBinCount(), hogSettings.insensitiveBinCount_
    );
}

void HogProcessor::calculateHogOcl()
{
    cl_event imageWriteEvent = NULL;
    size_t bytes = ocvImageGrayFloat_->rows * ocvImageGrayFloat_->cols * sizeof(cl_float);
    cl_int status = clEnqueueWriteBuffer(oclQueue_, hog_.image_, CL_FALSE, 0, bytes,
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
        hogProto_.settings_.sensitiveBinCount() * sizeof(cl_float);
    cl_uint *cellDescriptor = NULL;
    if (status == CL_SUCCESS)
    {
        cellDescriptor = (cl_uint*)clEnqueueMapBuffer(oclQueue_, hog_.cellDescriptor_, CL_TRUE,
            CL_MAP_READ, 0, bytes, 1, &hogEvent, NULL, &status);
    }
    if (hogEvent)
    {
        clReleaseEvent(hogEvent);
        hogEvent = NULL;
    }
    if (cellDescriptor)
    {
        compareDescriptorsOcl(cellDescriptor);
    }
    cl_event unmapEvent = NULL;
    if (cellDescriptor)
    {
        status = clEnqueueUnmapMemObject(oclQueue_, hog_.cellDescriptor_, cellDescriptor,
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

void HogProcessor::calculateHogPiotr()
{
    cv::Mat ocvImage(captureSettings_.frameHeight_, captureSettings_.frameWidth_, CV_8UC3, rgbFrame_);
    cv::cvtColor(ocvImage, *ocvImageGray_, CV_RGB2GRAY);
    ocvImageGray_->convertTo(*ocvImageGrayFloat_, CV_32FC1);

    int croppedSize[2];
    int halfPadding[2];
    for (int i = 0; i < 2; ++i)
    {
        croppedSize[i] = hogProto_.settings_.cellCount_[i] * hogProto_.settings_.cellSize_;
        halfPadding[i] = hogProto_.settings_.halfPadding_[i];
    }
    cv::Mat_<float> ocvImGrayCropped = (*ocvImageGrayFloat_)(cv::Rect(
        halfPadding[0], halfPadding[1], croppedSize[0], croppedSize[1]));

    float pixelMax = std::numeric_limits<float>::lowest();
    float pixelMin = std::numeric_limits<float>::max();
    for (int y = 0; y < croppedSize[1]; ++y)
    {
        for (int x = 0; x < croppedSize[0]; ++x)
        {
            float pixel = ocvImGrayCropped.at<float>(y, x);
            pixelMax = fmaxf(pixelMax, pixel);
            pixelMin = fminf(pixelMin, pixel);
        }
    }
    std::cout << "pixelMax = " << pixelMax << "   pixelMin = " << pixelMin << "\n";

    const HogSettings &hogSettings = hogProto_.settings_;
    std::vector<cv::Mat> hogPiotr = FHoG::extract(
        ocvImGrayCropped, 2, hogSettings.cellSize_, hogSettings.insensitiveBinCount_,
        1, hogSettings.truncation_);
    for (int c = 0; c < hogSettings.channelsPerFeature(); ++c)
    {
        const cv::Mat &hogChannel = hogPiotr[c];
        for (int y = 0; y < hogSettings.cellCount_[1]; ++y)
        {
            for (int x = 0; x < hogSettings.cellCount_[0]; ++x)
            {
                int featureIndex = x + y * hogSettings.cellCount_[0];
                hogPiotr_[featureIndex * hogSettings.channelsPerFeature() + c] =
                    hogChannel.at<float>(y, x);
            }
        }
    }
}

void HogProcessor::compareDescriptors() const
{
    const HogSettings &hogSettings = hogProto_.settings_;
    int cellCount[2] = { hogSettings.cellCount_[0], hogSettings.cellCount_[1] };
    int cellCountTotal = cellCount[0] * cellCount[1];
    int channelsPerFeature = hogSettings.channelsPerFeature();

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
                continue;
            }

            int channelIndex = c * channelsPerFeature + b;
            float ours = hogProto_.featureDescriptor_[channelIndex];
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

void HogProcessor::compareDescriptorsOcl(const uint *mappedDescriptor) const
{
    const HogSettings &hogSettings = hogProto_.settings_;
    int cellCount[2] = { hogSettings.cellCount_[0], hogSettings.cellCount_[1] };
    int channelsPerCell = hogSettings.channelsPerCell();
    int sensitiveBinCount = hogSettings.sensitiveBinCount();

    int mismatchCount = 0;
    int nonZerosCount = 0;
    for (int y = 0; y < cellCount[1]; ++y)
    {
        for (int x = 0; x < cellCount[0]; ++x)
        {
            for (int b = 0; b < sensitiveBinCount; ++b)
            {
                int c = x + y * cellCount[0];
                float ours = (float)mappedDescriptor[c * sensitiveBinCount + b] * 1e-6f;
                float gt = hogProto_.cellDescriptor_[c * channelsPerCell + b];
                float delta = gt - ours;
                if (fabsf(delta) > fmaxf(gt, 1e-3f) * 0.01f)
                {
                    mismatchCount++;
//                    std::cout << "(" << x << ", " << y << ")\n";
                }
                nonZerosCount += fabsf(ours) > 1e-3f;
            }
        }
//        std::cout << y << ": " << mismatchCount << "\n";
    }

    int channelsTotal = cellCount[0] * cellCount[1] * sensitiveBinCount;
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

void HogProcessor::processFrame()
{
    if (!captureFrame())
    {
        qDebug("Failed to capture %d-th frame", frameIndex_);
    }

    calculateHogPiotr();
    hogProto_.calculate((float*)ocvImageGrayFloat_->data);
    calculateHogOcl();
    compareDescriptors();

    QImage qimage(rgbFrame_, captureSettings_.frameWidth_, captureSettings_.frameHeight_,
        captureSettings_.frameWidth_ * 3, QImage::Format_RGB888);
    emit sendFrame(qimage.copy());

    const HogSettings &s = hogProto_.settings_;
    QVector<float> container(s.cellCount_[0] * s.cellCount_[1] * s.channelsPerFeature(), 0.0f);
    const float *desc = hogProto_.featureDescriptor_; // hogPiotr_;
    qCopy(desc, desc + container.size(), container.begin());
    emit sendHog(container);
}

