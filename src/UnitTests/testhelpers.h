#ifndef TESTHELPERS_H
#define TESTHELPERS_H

#include <QImage>

struct TestImageSettings
{
    const int width_ = 1280;
    const int height_ = 720;
    const QString path_ = "C:/tracking/data/soldier/00000138.jpg";
};

inline QImage loadTestImage()
{
    TestImageSettings s;
    return QImage(s.path_).convertToFormat(QImage::Format_RGB888);
}

#endif // TESTHELPERS_H
