#include <QSettings>
#include <QFileDialog>
#include <hogmainwin.h>
#include <ui_hogmainwin.h>

HogMainWin::HogMainWin(QWidget *parent)
    : VideoCaptureBase(parent)
    , ui_(new Ui::HogMainWin)
    , hogProcessor_(new HogProcessor)
{
    ui_->setupUi(this);
    playPauseBtn_ = ui_->playPauseBtn;
    connectBaseControls(ui_->openDirBtn, hogProcessor_);
    connect(hogProcessor_, SIGNAL(sendFrame(QImage)),
            ui_->videoWidget, SLOT(setFrame(QImage)));
    connect(hogProcessor_, SIGNAL(sendHog(QVector<float>)),
            ui_->hogWidget, SLOT(setHog(QVector<float>)));
    connect(hogProcessor_, SIGNAL(sendHogSettings(int,int,int,int,int)),
            ui_->hogWidget, SLOT(setUp(int,int,int,int,int)));
    emit sendVideoCaptureState(VideoProcessor::CaptureState::NotInitialized);
}

HogMainWin::~HogMainWin()
{
    if (ui_)
    {
        delete ui_;
        ui_ = nullptr;
    }
    if (hogProcessor_)
    {
        delete hogProcessor_;
        hogProcessor_ = nullptr;
    }
}

