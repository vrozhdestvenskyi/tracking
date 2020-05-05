#include <hogprocessor.h>
#include <opencv2/opencv.hpp>
#include <QImage>
#include <QVector>

HogProcessor::HogProcessor(QObject *parent)
    : VideoProcessor(parent)
    , ocvImageGray_(new cv::Mat_<uchar>())
    , ocvImageGrayFloat_(new cv::Mat_<float>())
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
    if (oclImage_)
    {
        clReleaseMemObject(oclImage_);
        oclImage_ = NULL;
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

    if (!hogSett_.init(settings.frameWidth_, settings.frameHeight_))
    {
        return emitError("Invalid image resolution passed into HogSettings");
    }
    int bytes = hogSett_.imWidth() * hogSett_.imHeight() * sizeof(cl_float);
    oclImage_ = clCreateBuffer(oclContext_, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    if (!oclImage_)
    {
        return emitError("Failed to initialize oclImage_");
    }
    if (hog_.initialize(hogSett_, oclContext_, oclProgram_, oclImage_) != CL_SUCCESS)
    {
        return emitError("Failed to initialize Hog");
    }
    desc_ = new float [hogSett_.descLen()];
    msSum_ = 0;
    emit sendHogSettings(
        hogSett_.cellCount_[0], hogSett_.cellCount_[1], hogSett_.channelsPerBlock(),
        0, hogSett_.sensitiveBinCount()
//        hogSett_.sensitiveBinCount(), hogSett_.insensitiveBinCount_
    );
    return true;
}

void HogProcessor::calcHog()
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
    bytes = hogSett_.descLen() * sizeof(cl_float);
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
    cl_event unmapEvent = NULL;
    if (mappedDesc)
    {
        std::copy(mappedDesc, mappedDesc + hogSett_.descLen(), desc_);
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

bool HogProcessor::processFrame()
{
    if (!captureFrame())
    {
        qDebug("Failed to capture %d-th frame", frameIndex_);
        return false;
    }
    cv::Mat ocvImage(captureSettings_.frameHeight_, captureSettings_.frameWidth_, CV_8UC3, rgbFrame_);
    cv::cvtColor(ocvImage, *ocvImageGray_, CV_RGB2GRAY);
    ocvImageGray_->convertTo(*ocvImageGrayFloat_, CV_32FC1);
    calcHog();

    QImage qimage(rgbFrame_, captureSettings_.frameWidth_, captureSettings_.frameHeight_,
        captureSettings_.frameWidth_ * 3, QImage::Format_RGB888);
    emit sendFrame(qimage.copy());

    QVector<float> container(hogSett_.descLen(), 0.0f);
    qCopy(desc_, desc_ + container.size(), container.begin());
    emit sendHog(container);
    return true;
}

