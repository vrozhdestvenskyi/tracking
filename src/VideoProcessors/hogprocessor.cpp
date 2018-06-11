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
{}

HogProcessor::~HogProcessor()
{
    release();
}

void HogProcessor::release()
{
    releaseHogHandle(hogHandle_);
    if (hogPiotr_)
    {
        delete [] hogPiotr_;
        hogPiotr_ = nullptr;
    }
}

void HogProcessor::setupProcessor(const VideoProcessor::CaptureSettings &settings)
{
    release();

    HogSettings hogSettings;
    hogSettings.cellSize_ = 4;
    hogSettings.insensitiveBinCount_ = 9;
    hogSettings.truncation_ = 0.2f;

    if (settings.frameWidth_ % hogSettings.cellSize_ ||
        settings.frameHeight_ % hogSettings.cellSize_)
    {
        throw std::runtime_error("Image resolution is not a multiple of HOG cell size");
    }
    hogSettings.cellCount_[0] = settings.frameWidth_ / hogSettings.cellSize_;
    hogSettings.cellCount_[1] = settings.frameHeight_ / hogSettings.cellSize_;

    initializeHogHandle(hogSettings, hogHandle_);

    int cellCount = hogSettings.cellCount_[0] * hogSettings.cellCount_[1];
    int length = cellCount * hogSettings.channelsPerFeature();
    hogPiotr_ = new float [length];
    std::fill(hogPiotr_, hogPiotr_ + length, 0.0f);

    VideoProcessor::setupProcessor(settings);

    emit sendHogSettings(
        hogSettings.cellCount_[0], hogSettings.cellCount_[1], hogSettings.channelsPerFeature(),
        0, hogSettings.sensitiveBinCount()
//        hogSettings.sensitiveBinCount(), hogSettings.insensitiveBinCount_
    );
}

void HogProcessor::calculateHogPiotr()
{
    cv::Mat ocvImage(
        captureSettings_.frameHeight_, captureSettings_.frameWidth_, CV_8UC3, rgbFrame_
    );
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

    const HogSettings &hogSettings = hogHandle_.settings_;
    std::vector<cv::Mat> hogPiotr = FHoG::extract(
        *ocvImageGrayFloat_, 2, hogSettings.cellSize_, hogSettings.insensitiveBinCount_,
        1, hogSettings.truncation_
    );
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
                float v = hogHandle_.featureDescriptor_[featureIndex * hogSettings.channelsPerFeature() + c];
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
    const HogSettings &hogSettings = hogHandle_.settings_;
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
            float ours = hogHandle_.featureDescriptor_[channelIndex];
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

void HogProcessor::processFrame()
{
    if (!captureFrame())
    {
        qDebug("Failed to capture %d-th frame", frameIndex_);
    }

    calculateHogPiotr();
    calculateHog(ocvImageGray_->data, hogHandle_);
    compareDescriptors();

    QImage qimage(
        rgbFrame_, captureSettings_.frameWidth_, captureSettings_.frameHeight_,
        captureSettings_.frameWidth_ * 3, QImage::Format_RGB888
    );
    emit sendFrame(qimage.copy());

    const HogSettings &hogSettings = hogHandle_.settings_;
    QVector<float> hogContainer(
        hogSettings.cellCount_[0] * hogSettings.cellCount_[1] * hogSettings.channelsPerFeature()
    );
//    qCopy(hogPiotr_, hogPiotr_ + hogContainer.size(), hogContainer.begin());
    qCopy(hogHandle_.featureDescriptor_, hogHandle_.featureDescriptor_ + hogContainer.size(), hogContainer.begin());
    emit sendHog(hogContainer);
}
