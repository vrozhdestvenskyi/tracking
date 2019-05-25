#include <iostream>
#include <QImage>
#include <opencv2/opencv.hpp>
#include <labproto.h>

void countMismatches(const QImage &src, const QImage &dst, int delta, const std::string &title)
{
    if (src.width() != dst.width() || src.height() != dst.height() ||
        src.format() != dst.format())
    {
        throw std::runtime_error("Expected to get images of the same resolutions and formats");
    }
    int sz = src.width() * src.height();
    float vMax[3], vMin[3];
    for (int i = 0; i < 3; ++i)
    {
        vMin[i] = FLT_MAX;
        vMax[i] = -vMin[i];
    }
    int cnt[3] = { 0, 0, 0 };
    for (int pix = 0; pix < sz; ++pix)
    {
        for (int c = 0; c < 3; ++c)
        {
            const int i = pix * 3 + c;
            cnt[c] += std::abs(src.bits()[i] - dst.bits()[i]) > delta;
            float f = src.bits()[i];
            vMin[c] = fminf(vMin[c], f);
            vMax[c] = fmaxf(vMax[c], f);
        }
    }
    std::cout << "min: " << vMin[0] << ", " << vMin[1] << ", " << vMin[2] << "\n";
    std::cout << "max: " << vMax[0] << ", " << vMax[1] << ", " << vMax[2] << "\n";
    std::cout << title << ": ";
    std::cout << "mismatched [" << cnt[0] << ", " << cnt[1] << ", " << cnt[2] << "] values";
    std::cout << "(";
    for (int c = 0; c < 3; ++c)
    {
        std::cout << (cnt[c] * 100.0f / sz) << "%" << (c + 1 < 3 ? ", " : "");
    }
    std::cout << ") from " << sz << "\n";
}

void testLab()
{
    const QString path("C:/tracking/data/fernando/00000001.jpg");
    std::cout << std::string(path.toLocal8Bit()) << "\n";

    const QImage im(QImage(path).convertToFormat(QImage::Format_RGB888));
    const int sz = im.width() * im.height();

    QImage gtLab(im.width(), im.height(), QImage::Format_RGB888);
    QImage gtRgb(im.width(), im.height(), QImage::Format_RGB888);
    {
        cv::Mat ocvIm(im.height(), im.width(), CV_8UC3, (void*)im.bits());
        cv::Mat ocvLab(im.height(), im.width(), CV_8UC3);
        cv::cvtColor(ocvIm, ocvLab, CV_RGB2Lab);
        std::cout << ocvLab.type() << " " << CV_8UC3 << "\n";
        int bytes = sz * 3 * sizeof(uchar);
        std::copy(ocvLab.data, ocvLab.data + bytes, gtLab.bits());
        cv::Mat ocvRgb(im.height(), im.width(), CV_8UC3);
        cv::cvtColor(ocvLab, ocvRgb, CV_Lab2RGB);
        std::copy(ocvRgb.data, ocvRgb.data + bytes, gtRgb.bits());
//        std::copy(ocvIm.data, ocvIm.data + sz * 3 * sizeof(uchar), gtgb.bits());
    }

    QImage lab(im.width(), im.height(), QImage::Format_RGB888);
    QImage rgb(im.width(), im.height(), QImage::Format_RGB888);
    rgb2lab(im.bits(), sz, lab.bits());
    lab2rgb(lab.bits(), sz, rgb.bits());

    int delta = 3;
    countMismatches(lab, gtLab, delta, "RGB-LAB(ours-ocv)");
    countMismatches(rgb, gtRgb, delta, "RGB-LAB-RGB(ours-ocv)");
    countMismatches(rgb, im, delta, "RGB-LAB-RGB(ours-raw)");
    countMismatches(im, gtRgb, delta, "RGB-LAB-RGB(ocv-raw)");
}

int main(int argc, char *argv[])
{
    testLab();
    return 0;
}

