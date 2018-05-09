#include <QSettings>
#include <QFileDialog>
#include <videocapturemainwin.h>
#include <ui_videocapturemainwin.h>

VideoCaptureMainWin::VideoCaptureMainWin(QWidget *parent)
    : VideoCaptureBase(parent)
    , ui_(new Ui::VideoCaptureMainWin)
    , videoProcessor_(new VideoProcessor)
{
    ui_->setupUi(this);
    playPauseBtn_ = ui_->playPauseBtn;
    connectBaseControls(ui_->openDirBtn, videoProcessor_);
    connect(videoProcessor_, SIGNAL(sendFrame(QImage)), ui_->videoWidget, SLOT(setFrame(QImage)));
    emit sendVideoCaptureState(VideoProcessor::CaptureState::NotInitialized);
}

VideoCaptureMainWin::~VideoCaptureMainWin()
{
    if (ui_)
    {
        delete ui_;
        ui_ = nullptr;
    }
    if (videoProcessor_)
    {
        delete videoProcessor_;
        videoProcessor_ = nullptr;
    }
}
