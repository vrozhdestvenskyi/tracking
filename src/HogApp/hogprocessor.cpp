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

    HogSettings hogSettings;
    if (settings.frameWidth_ % hogSettings.cellSize_ ||
        settings.frameHeight_ % hogSettings.cellSize_)
    {
        emitError("Image resolution is not a multiple of HOG cell size");
        return;
    }
    hogSettings.cellCount_[0] = settings.frameWidth_ / hogSettings.cellSize_;
    hogSettings.cellCount_[1] = settings.frameHeight_ / hogSettings.cellSize_;
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
    cl_float *cellDescriptor = NULL;
    if (status == CL_SUCCESS)
    {
        cellDescriptor = (cl_float*)clEnqueueMapBuffer(oclQueue_, hog_.cellDescriptor_, CL_TRUE,
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

    float pixelMax = std::numeric_limits<float>::lowest();
    float pixelMin = std::numeric_limits<float>::max();
    for (int y = 0; y < ocvImageGrayFloat_->rows; ++y)
    {
        for (int x = 0; x < ocvImageGrayFloat_->cols; ++x)
        {
            float pixel = ocvImageGrayFloat_->at<float>(y, x);
            pixelMax = fmaxf(pixelMax, pixel);
            pixelMin = fminf(pixelMin, pixel);
        }
    }
//    std::cout << "pixelMax = " << pixelMax << "   pixelMin = " << pixelMin << "\n";

    const HogSettings &hogSettings = hogProto_.settings_;
    std::vector<cv::Mat> hogPiotr = FHoG::extract(
        *ocvImageGrayFloat_, 2, hogSettings.cellSize_, hogSettings.insensitiveBinCount_,
        1, hogSettings.truncation_);
    for (int c = 0; c < hogSettings.channelsPerFeature(); ++c)
    {
        // debug
        float channelMax = std::numeric_limits<float>::lowest();
        float channelMin = std::numeric_limits<float>::max();
        float channelSum = 0.0f;
        float channelSumSquares = 0.0f;

        const cv::Mat &hogChannel = hogPiotr[c];
        for (int y = 0; y < hogSettings.cellCount_[1]; ++y)
        {
            for (int x = 0; x < hogSettings.cellCount_[0]; ++x)
            {
                int featureIndex = x + y * hogSettings.cellCount_[0];
                hogPiotr_[featureIndex * hogSettings.channelsPerFeature() + c] =
                    hogChannel.at<float>(y, x);

                // debug
//                float v = hogPiotr_[featureIndex * hogSettings.channelsPerFeature() + c];
                float v = hogProto_.featureDescriptor_[featureIndex * hogSettings.channelsPerFeature() + c];
                channelMax = std::max<float>(channelMax, v);
                channelMin = std::min<float>(channelMin, v);
                channelSum += v;
                channelSumSquares += v * v;
            }
        }

        // debug
//        int n = hogSettings.cellCount_[0] * hogSettings.cellCount_[1];
//        std::cout << std::setprecision(3) << std::fixed;
//        std::cout << "chnl:" << c << " min:" << channelMin << " max:" << channelMax
//            << " mean:" << channelSum / n << " std:"
//            << sqrtf(channelSumSquares / n - std::pow(channelSum / n, 2)) << "\n";
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
            if (cellX == 0 || cellX == cellCount[0] - 1 ||
                cellY == 0 || cellY == cellCount[1] - 1)
            {
                //continue;
            }

            int channelIndex = c * channelsPerFeature + b;
            float ours = hogProto_.featureDescriptor_[channelIndex];
            float gt = hogPiotr_[channelIndex];
            float delta = gt - ours;
            if (fabsf(delta) > gt * 0.05f)
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
    int cellCountTotal = cellCount[0] * cellCount[1];
    int channelsPerCell = hogSettings.channelsPerCell();
    int sensitiveBinCount = hogSettings.sensitiveBinCount();

    int mismatchCount = 0;
    int nonZerosCount = 0;
    for (int c = 0; c < cellCountTotal; ++c)
    {
        for (int b = 0; b < sensitiveBinCount; ++b)
        {
            int cellX = c % cellCount[0];
            int cellY = c / cellCount[0];
            if (cellX == 0 || cellX == cellCount[0] - 1 ||
                cellY == 0 || cellY == cellCount[1] - 1)
            {
                //continue;
            }

            float ours = mappedDescriptor[c * sensitiveBinCount + b];
            float gt = hogProto_.cellDescriptor_[c * channelsPerCell + b];
            float delta = gt - ours;
            if (fabsf(delta) > gt * 0.05f)
            {
                mismatchCount++;
            }
            nonZerosCount += fabsf(ours) > 1e-3f;
        }
    }

    int channelsTotal = cellCountTotal * sensitiveBinCount;
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
    hogProto_.calculate(ocvImageGray_->data);
    calculateHogOcl();
//    compareDescriptors();

    QImage qimage(rgbFrame_, captureSettings_.frameWidth_, captureSettings_.frameHeight_,
        captureSettings_.frameWidth_ * 3, QImage::Format_RGB888);
    emit sendFrame(qimage.copy());

    const HogSettings &hogSettings = hogProto_.settings_;
    QVector<float> hogContainer(
        hogSettings.cellCount_[0] * hogSettings.cellCount_[1] * hogSettings.channelsPerFeature());
//    qCopy(hogPiotr_, hogPiotr_ + hogContainer.size(), hogContainer.begin());
    qCopy(hogProto_.featureDescriptor_, hogProto_.featureDescriptor_ + hogContainer.size(),
        hogContainer.begin());
    emit sendHog(hogContainer);
}

