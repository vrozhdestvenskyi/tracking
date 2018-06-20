#ifndef VIDEOCAPTUREBASE_H
#define VIDEOCAPTUREBASE_H

#include <QMainWindow>
#include <QPushButton>
#include <videoprocessor.h>

class VideoCaptureBase : public QMainWindow
{
    Q_OBJECT

public:
    virtual ~VideoCaptureBase();

protected:
    VideoCaptureBase(QWidget *parent = nullptr);
    void connectBaseControls(
        const QPushButton *openDirBtn,
        const VideoProcessor *videoProcessor) const;

protected slots:
    void receiveError(const QString &what);
    void openDirPressed();
    void playPausePressed();
    void setVideoCaptureState(VideoProcessor::CaptureState state);

signals:
    void setupProcessor(const VideoProcessor::CaptureSettings &settings);
    void sendVideoCaptureState(VideoProcessor::CaptureState state);

protected:
    bool setupVideoDir(const QString &directory, VideoProcessor::CaptureSettings &settings) const;

    const QString videoSourceSettingsPath_ = "videoSourceSettings.ini";
    const QStringList imageExtensions_ = { "jpg", "JPG" };
    const QString btnTextPaused_ = "Paused";
    const QString btnTextCapturing_ = "Capturing";
    QPushButton *playPauseBtn_ = nullptr;
};

#endif // VIDEOCAPTUREBASE_H
