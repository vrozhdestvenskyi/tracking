#include <videocapturemainwin.h>
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    qRegisterMetaType<VideoProcessor::CaptureSettings>();
    qRegisterMetaType<VideoProcessor::CaptureState>();
    VideoCaptureMainWin win;
    win.show();
    return app.exec();
}
