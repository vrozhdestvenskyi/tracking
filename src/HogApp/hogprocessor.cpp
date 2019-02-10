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
    blockHog_.release();
    invBlockNorm_.release();
    cellNormSumX_.release();
    cellNorm_.release();
    hog_.release();
    derivs_.release();
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
    if (!hogSettings.init(settings.frameWidth_, settings.frameHeight_))
    {
        emitError("Invalid image resolution passed into HogSettings");
        return;
    }
    hogProto_.initialize(hogSettings);
    {
        int bytes = hogSettings.imWidth() * hogSettings.imHeight() * sizeof(cl_float);
        oclImage_ = clCreateBuffer(oclContext_, CL_MEM_READ_ONLY, bytes, NULL, NULL);
        if (!oclImage_)
        {
            emitError("Failed to initialize oclImage_");
            return;
        }
    }
    if (derivs_.initialize(hogSettings, oclContext_, oclProgram_, oclImage_) != CL_SUCCESS)
    {
        emitError("Failed to initialize Derivs");
        return;
    }
    if (hog_.initialize(hogSettings, oclContext_, oclProgram_, derivs_.derivsX_,
            derivs_.derivsY_) != CL_SUCCESS)
    {
        emitError("Failed to initialize CellHog");
        return;
    }
    if (cellNorm_.initialize(hogSettings, oclContext_, oclProgram_, hog_.descriptor_) != CL_SUCCESS)
    {
        emitError("Failed to initialize CellNorm");
        return;
    }
    if (cellNormSumX_.initialize(hogSettings, cellNorm_.padding_, oclContext_, oclProgram_,
            cellNorm_.cellNorms_) != CL_SUCCESS)
    {
        emitError("Failed to initialize CellNormSumX");
        return;
    }
    if (invBlockNorm_.initialize(hogSettings, cellNorm_.padding_, oclContext_, oclProgram_,
            cellNormSumX_.normSums_) != CL_SUCCESS)
    {
        emitError("Failed to initialize InvBlockNorm");
        return;
    }
    if (blockHog_.initialize(hogSettings, cellNorm_.padding_, oclContext_, oclProgram_,
            hog_.descriptor_, invBlockNorm_.invBlockNorms_) != CL_SUCCESS)
    {
        emitError("Failed to initialize BlockHog");
        return;
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
}

void HogProcessor::calculateHogOcl()
{
    timer_.restart();
    cl_event imageWriteEvent = NULL;
    size_t bytes = ocvImageGrayFloat_->rows * ocvImageGrayFloat_->cols * sizeof(cl_float);
    cl_int status = clEnqueueWriteBuffer(oclQueue_, oclImage_, CL_FALSE, 0, bytes,
        ocvImageGrayFloat_->data, 0, NULL, &imageWriteEvent);
    cl_event derivsEvent = NULL;
    if (status == CL_SUCCESS)
    {
        status = derivs_.calculate(oclQueue_, 1, &imageWriteEvent, derivsEvent);
    }
    if (imageWriteEvent)
    {
        clReleaseEvent(imageWriteEvent);
        imageWriteEvent = NULL;
    }
    cl_event cellHogEvent = NULL;
    if (status == CL_SUCCESS)
    {
        status = hog_.calculate(oclQueue_, 1, &derivsEvent, cellHogEvent);
    }
    if (derivsEvent)
    {
        clReleaseEvent(derivsEvent);
        derivsEvent = NULL;
    }
    cl_event cellNormEvent = NULL;
    if (status == CL_SUCCESS)
    {
        status = cellNorm_.calculate(oclQueue_, 1, &cellHogEvent, cellNormEvent);
    }
    if (cellHogEvent)
    {
        clReleaseEvent(cellHogEvent);
        cellHogEvent = NULL;
    }
    cl_event sumXevent = NULL;
    if (status == CL_SUCCESS)
    {
        status = cellNormSumX_.calculate(oclQueue_, 1, &cellNormEvent, sumXevent);
    }
    if (cellNormEvent)
    {
        clReleaseEvent(cellNormEvent);
        cellNormEvent = NULL;
    }
    cl_event blockNormEvent = NULL;
    if (status == CL_SUCCESS)
    {
        status = invBlockNorm_.calculate(oclQueue_, 1, &sumXevent, blockNormEvent);
    }
    if (sumXevent)
    {
        clReleaseEvent(sumXevent);
        sumXevent = NULL;
    }
    cl_event blockHogEvent = NULL;
    if (status == CL_SUCCESS)
    {
        status = blockHog_.calculate(oclQueue_, 1, &blockNormEvent, blockHogEvent);
    }
    if (blockNormEvent)
    {
        clReleaseEvent(blockNormEvent);
        blockNormEvent = NULL;
    }
    bytes = hogProto_.settings_.cellCount_[0] * hogProto_.settings_.cellCount_[1] *
        hogProto_.settings_.channelsPerBlock() * sizeof(cl_float);
    cl_float *mappedDesc = NULL;
    if (status == CL_SUCCESS)
    {
        mappedDesc = (cl_float*)clEnqueueMapBuffer(oclQueue_, blockHog_.descriptor_, CL_TRUE,
            CL_MAP_READ, 0, bytes, 1, &blockHogEvent, NULL, &status);
    }
    quint64 ms = timer_.restart();
    msSum_ += ms;
    std::cout << frameIndex_ << ": " << ms << "ms\n";
    if (blockHogEvent)
    {
        clReleaseEvent(blockHogEvent);
        blockHogEvent = NULL;
    }
    if (mappedDesc)
    {
        compareDescriptorsOcl(mappedDesc);
        //compareDescriptors(mappedDesc);
    }
    cl_event unmapEvent = NULL;
    if (mappedDesc)
    {
        status = clEnqueueUnmapMemObject(oclQueue_, blockHog_.descriptor_, mappedDesc,
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

    const HogSettings &hogSettings = hogProto_.settings_;
    std::vector<cv::Mat> hogPiotr = FHoG::extract(
        *ocvImageGray_, 2, hogSettings.cellSize_, hogSettings.insensitiveBinCount_,
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

void HogProcessor::processFrame()
{
    if (!captureFrame())
    {
        qDebug("Failed to capture %d-th frame", frameIndex_);
    }

    calculateHogPiotr();
    hogProto_.calculate((float*)ocvImageGrayFloat_->data);
    calculateHogOcl();
    compareDescriptors(hogProto_.blockDescriptor_);

    QImage qimage(rgbFrame_, captureSettings_.frameWidth_, captureSettings_.frameHeight_,
        captureSettings_.frameWidth_ * 3, QImage::Format_RGB888);
    emit sendFrame(qimage.copy());

    const HogSettings &s = hogProto_.settings_;
    QVector<float> container(s.cellCount_[0] * s.cellCount_[1] * s.channelsPerBlock(), 0.0f);
    const float *desc = hogProto_.blockDescriptor_; // hogPiotr_;
    qCopy(desc, desc + container.size(), container.begin());
    emit sendHog(container);
}

