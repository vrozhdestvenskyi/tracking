#include <hogmainwin.h>
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    qRegisterMetaType<VideoProcessor::CaptureSettings>();
    qRegisterMetaType<VideoProcessor::CaptureState>();
    HogMainWin win;
    win.show();
    return app.exec();
}
