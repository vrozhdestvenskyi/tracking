#ifndef VIDEOCAPTUREMAINWIN_H
#define VIDEOCAPTUREMAINWIN_H

#include <videoprocessor.h>
#include <videocapturebase.h>

namespace Ui
{
    class VideoCaptureMainWin;
}

class VideoCaptureMainWin : public VideoCaptureBase
{
    Q_OBJECT

public:
    explicit VideoCaptureMainWin(QWidget *parent = nullptr);
    virtual ~VideoCaptureMainWin();

protected:
    Ui::VideoCaptureMainWin *ui_ = nullptr;
    VideoProcessor *videoProcessor_ = nullptr;
};

#endif // VIDEOCAPTUREMAINWIN_H
