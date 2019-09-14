#include <videoprocessor.h>
#include <QImage>
#include <QMetaMethod>
#include <QImageReader>

VideoProcessor::VideoProcessor(QObject *parent)
    : QObject(parent)
    , OclProcessor()
    , captureTimer_(this)
{
    connect(&captureTimer_, SIGNAL(timeout()), this, SLOT(processFrame()));
}

void VideoProcessor::release()
{
    if (rgbFrame_)
    {
        delete [] rgbFrame_;
        rgbFrame_ = nullptr;
    }
    OclProcessor::release();
}

VideoProcessor::~VideoProcessor()
{
    release();
}

bool VideoProcessor::setupProcessor(const VideoProcessor::CaptureSettings &settings)
{
    release();
    if (OclProcessor::initialize() != CL_SUCCESS)
    {
        setVideoCaptureState(CaptureState::NotInitialized);
        emit sendError("OclProcessor::initialize() was failed");
        return false;
    }
    captureSettings_ = settings;
    if (settings.frameWidth_ <= 0 || settings.frameHeight_ <= 0)
    {
        setVideoCaptureState(CaptureState::NotInitialized);
        emit sendError("VideoProcessor::setupProcessor() received invalid frame size");
        return false;
    }
    int dataLength = settings.frameWidth_ * settings.frameHeight_ * 3 * sizeof(uchar);
    rgbFrame_ = new uchar [dataLength];
    std::fill(rgbFrame_, rgbFrame_ + dataLength, 0);
    frameIndex_ = 0;
    setVideoCaptureState(CaptureState::Paused);
    QImage qimage(rgbFrame_, settings.frameWidth_, settings.frameHeight_,
        settings.frameWidth_ * 3, QImage::Format_RGB888);
    emit sendFrame(qimage.copy());
    return true;
}

bool VideoProcessor::processFrame()
{
    qDebug("VideoProcessor::processFrame()");
    if (!captureFrame())
    {
        qDebug("Failed to capture %d-th frame", frameIndex_);
        return false;
    }
    QImage qimage(rgbFrame_, captureSettings_.frameWidth_, captureSettings_.frameHeight_,
        captureSettings_.frameWidth_ * 3, QImage::Format_RGB888);
    emit sendFrame(qimage.copy());
    return true;
}

void VideoProcessor::setVideoCaptureState(VideoProcessor::CaptureState state)
{
    switch (state)
    {
    case CaptureState::NotInitialized:
        captureTimer_.stop();
        break;
    case CaptureState::Paused:
        captureTimer_.stop();
        break;
    case CaptureState::Capturing:
        captureTimer_.start(40);
        break;
    default:
        throw std::runtime_error("VideoProcessor::setVideoCaptureState(...): invalid state");
    }
    emit sendVideoCaptureState(state);
}

bool VideoProcessor::captureFrame()
{
    switch (captureSettings_.captureMode_)
    {
    case CaptureMode::FromDirectory:
        return captureFrameFromDir();
    default:
        throw std::runtime_error("VideoProcessor::captureFrame(): invalid video capture mode");
    }
    return false;
}

bool VideoProcessor::captureFrameFromDir()
{
    const VideoDirectorySettings &settings = captureSettings_.videoDirSettings_;
    if (frameIndex_ >= settings.frameCount_)
    {
        qDebug("All video frames have been captured");
        setVideoCaptureState(CaptureState::NotInitialized);
        return false;
    }
    std::string frameNumber = std::to_string(settings.firstFrame_ + frameIndex_++);
    std::string leadingZeros(settings.digitCount_ - (int)frameNumber.length(), '0');
    std::string framePath = settings.directory_ + leadingZeros + frameNumber + settings.extension_;
    QImage qimage = QImage(QString(framePath.c_str())).convertToFormat(QImage::Format_RGB888);
    if (qimage.width() != captureSettings_.frameWidth_ ||
        qimage.height() != captureSettings_.frameHeight_ ||
        qimage.bytesPerLine() != captureSettings_.frameWidth_ * 3 ||
        qimage.format() != QImage::Format_RGB888)
    {
        setVideoCaptureState(CaptureState::NotInitialized);
        emit sendError("VideoProcessor::captureFrameFromDir(...): encountered an "
            "unexpected format of the captured frame");
        return false;
    }
    int frameSize = qimage.bytesPerLine() * qimage.height() * sizeof(uchar);
    std::copy(qimage.constBits(), qimage.constBits() + frameSize, rgbFrame_);
    return true;
}

