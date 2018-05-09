#ifndef HOGMAINWIN_H
#define HOGMAINWIN_H

#include <hogprocessor.h>
#include <videocapturebase.h>

namespace Ui
{
    class HogMainWin;
}

class HogMainWin : public VideoCaptureBase
{
    Q_OBJECT

public:
    explicit HogMainWin(QWidget *parent = nullptr);
    ~HogMainWin();

protected:
    Ui::HogMainWin *ui_ = nullptr;
    HogProcessor *hogProcessor_ = nullptr;
};

#endif // HOGMAINWIN_H
