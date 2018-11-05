#include <videocapturebase.h>
#include <QSettings>
#include <QFileDialog>
#include <QErrorMessage>

VideoCaptureBase::VideoCaptureBase(QWidget *parent)
    : QMainWindow(parent)
    , playPauseBtn_(nullptr)
{}

VideoCaptureBase::~VideoCaptureBase()
{}

void VideoCaptureBase::connectBaseControls(
    const QPushButton *openDirBtn, const VideoProcessor *videoProcessor) const
{
    if (!openDirBtn || !videoProcessor || !playPauseBtn_)
    {
        throw std::runtime_error("Can't connect signals/slots of nullptr");
    }
    connect(openDirBtn, SIGNAL(pressed()), this, SLOT(openDirPressed()));
    connect(playPauseBtn_, SIGNAL(pressed()), this, SLOT(playPausePressed()));
    connect(this, SIGNAL(setupProcessor(VideoProcessor::CaptureSettings)),
            videoProcessor, SLOT(setupProcessor(VideoProcessor::CaptureSettings)));
    connect(this, SIGNAL(sendVideoCaptureState(VideoProcessor::CaptureState)),
            videoProcessor, SLOT(setVideoCaptureState(VideoProcessor::CaptureState)));
    connect(videoProcessor, SIGNAL(sendVideoCaptureState(VideoProcessor::CaptureState)),
            this, SLOT(setVideoCaptureState(VideoProcessor::CaptureState)));
    connect(videoProcessor, SIGNAL(sendError(QString)), this, SLOT(receiveError(QString)));
}

void VideoCaptureBase::receiveError(const QString &what)
{
    QErrorMessage messageBox(this);
    messageBox.showMessage(what);
    messageBox.exec();
}

void VideoCaptureBase::openDirPressed()
{
    QSettings settings(videoSourceSettingsPath_, QSettings::IniFormat);
    QString directory = settings.value("directory", QString()).toString();

    directory = QFileDialog::getExistingDirectory(
        this, "Open directory with image sequence", directory, QFileDialog::ShowDirsOnly);
    if (!directory.length())
    {
        qDebug("Empty directory received!");
        return;
    }
    directory += QDir::separator();

    settings.setValue("directory", directory);
    settings.sync();

    VideoProcessor::CaptureSettings captureSettings;
    if (setupVideoDir(directory, captureSettings))
    {
        emit setupProcessor(captureSettings);
    }
}

void VideoCaptureBase::playPausePressed()
{
    if (!playPauseBtn_)
    {
        throw std::runtime_error("Play/pause button wasn't initialized");
    }
    QString btnTextCurrent = playPauseBtn_->text();
    if (btnTextCurrent == btnTextPaused_)
    {
        emit sendVideoCaptureState(VideoProcessor::CaptureState::Capturing);
    }
    else if (btnTextCurrent == btnTextCapturing_)
    {
        emit sendVideoCaptureState(VideoProcessor::CaptureState::Paused);
    }
    else
    {
        throw std::runtime_error("Invalid playPauseBtn state");
    }
}

void VideoCaptureBase::setVideoCaptureState(VideoProcessor::CaptureState state)
{
    if (!playPauseBtn_)
    {
        throw std::runtime_error("Play/pause button wasn't initialized");
    }
    QKeySequence shortcut = playPauseBtn_->shortcut();
    switch (state)
    {
    case VideoProcessor::CaptureState::NotInitialized:
        qDebug("VideoCaptureMainWin::setVideoCaptureState(NotInitialized)");
        playPauseBtn_->setEnabled(false);
        playPauseBtn_->setText(btnTextPaused_);
        playPauseBtn_->setShortcut(shortcut);
        break;
    case VideoProcessor::CaptureState::Paused:
        qDebug("VideoCaptureMainWin::setVideoCaptureState(Paused)");
        playPauseBtn_->setEnabled(true);
        playPauseBtn_->setText(btnTextPaused_);
        playPauseBtn_->setShortcut(shortcut);
        break;
    case VideoProcessor::CaptureState::Capturing:
        qDebug("VideoCaptureMainWin::setVideoCaptureState(Capturing)");
        playPauseBtn_->setEnabled(true);
        playPauseBtn_->setText(btnTextCapturing_);
        playPauseBtn_->setShortcut(shortcut);
        break;
    default:
        throw std::runtime_error("Invalid video capture state");
    }
}

bool VideoCaptureBase::setupVideoDir(
    const QString &directory, VideoProcessor::CaptureSettings &settings) const
{
    VideoProcessor::VideoDirectorySettings &dirSettings = settings.videoDirSettings_;
    settings.captureMode_ = VideoProcessor::CaptureMode::FromDirectory;
    dirSettings.directory_ = directory.toLocal8Bit().constData();

    QStringList fileNames = QDir(directory).entryList(QDir::Files, QDir::SortFlag::Name);
    for (const QString &fileName : fileNames)
    {
        QString extension = QFileInfo(fileName).suffix();
        if (!imageExtensions_.contains(extension))
        {
            continue;
        }
        QImage firstFrame;
        if (!firstFrame.load(directory + fileName))
        {
            continue;
        }
        settings.frameWidth_ = firstFrame.width();
        settings.frameHeight_ = firstFrame.height();
        dirSettings.extension_ = ("." + extension).toLocal8Bit().constData();
        dirSettings.digitCount_ = fileName.length() - extension.length() - 1;
        dirSettings.firstFrame_ = fileName.mid(0, dirSettings.digitCount_).toInt();
        break;
    }

    if (dirSettings.extension_.empty())
    {
        qDebug("setupVideoDir failed");
        return false;
    }

    dirSettings.frameCount_ = 0;
    for (const QString &fileName : fileNames)
    {
        QString extension = QFileInfo(fileName).suffix();
        dirSettings.frameCount_ += imageExtensions_.contains(extension);
    }
    return true;
}
