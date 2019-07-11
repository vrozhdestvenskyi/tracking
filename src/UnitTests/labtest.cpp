#include <iostream>
#include <QImage>
#include <opencv2/opencv.hpp>
#include <gtest/gtest.h>
#include <labproto.h>

void countMismatches(const QImage &src, const QImage &dst, const std::string &title)
{
    ASSERT_EQ(src.width(), dst.width());
    ASSERT_EQ(src.height(), dst.height());
    ASSERT_EQ(src.format(), dst.format());
    ASSERT_EQ(src.format(), QImage::Format_RGB888);

    const int pixDiffThr = 3;
    const float mismatchRatioThr = 1e-3f;
    const int sz = src.width() * src.height();
    int cnt[3] = { 0, 0, 0 };
    for (int pix = 0; pix < sz; ++pix)
    {
        for (int c = 0; c < 3; ++c)
        {
            const int i = pix * 3 + c;
            cnt[c] += std::abs((int)src.bits()[i] - (int)dst.bits()[i]) > pixDiffThr;
        }
    }
    std::cout << title << ": ";
    std::cout << "mismatched [" << cnt[0] << ", " << cnt[1] << ", " << cnt[2] << "] values ";
    std::cout << "(";
    for (int c = 0; c < 3; ++c)
    {
        std::cout << (cnt[c] * 100.0f / sz) << "%" << (c + 1 < 3 ? ", " : "");
    }
    std::cout << ") from " << sz << "\n";
    for (int c = 0; c < 3; ++c)
    {
        ASSERT_LT(cnt[c] / sz, mismatchRatioThr);
    }
}

TEST(LabTest, realData)
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

    countMismatches(lab, gtLab, "RGB-LAB(ours-ocv)");
    countMismatches(rgb, gtRgb, "RGB-LAB-RGB(ours-ocv)");
    countMismatches(rgb, im, "RGB-LAB-RGB(ours-raw)");
    countMismatches(im, gtRgb, "RGB-LAB-RGB(ocv-raw)");
}

