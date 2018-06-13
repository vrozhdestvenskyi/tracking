#ifndef VIDEOPROCESSOR_H
#define VIDEOPROCESSOR_H

#include <QTimer>

class VideoProcessor : public QObject
{
    Q_OBJECT

public:
    enum class CaptureMode
    {
        FromDirectory = 0
    };

    enum class CaptureState
    {
        NotInitialized = 0,
        Paused,
        Capturing
    };

    struct VideoDirectorySettings
    {
        int firstFrame_ = 0;
        int frameCount_ = 0;
        int digitCount_ = 0;
        std::string directory_;
        std::string extension_;
    };

    struct CaptureSettings
    {
        CaptureMode captureMode_ = (CaptureMode)0;
        int frameWidth_ = 0;
        int frameHeight_ = 0;
        VideoDirectorySettings videoDirSettings_;
    };

    VideoProcessor(QObject *parent = nullptr);
    virtual ~VideoProcessor();

public slots:
    virtual void setupProcessor(const VideoProcessor::CaptureSettings &settings);
    virtual void processFrame();
    virtual void setVideoCaptureState(VideoProcessor::CaptureState state);

signals:
    void sendVideoCaptureState(VideoProcessor::CaptureState state);
    void sendFrame(const QImage &image);

protected:
    bool captureFrame();
    bool captureFrameFromDir();

    CaptureSettings captureSettings_;
    uchar *rgbFrame_ = nullptr;
    int frameIndex_ = 0;
    QTimer captureTimer_;
};

Q_DECLARE_METATYPE(VideoProcessor::CaptureSettings)
Q_DECLARE_METATYPE(VideoProcessor::CaptureState)

#endif // VIDEOPROCESSOR_H
