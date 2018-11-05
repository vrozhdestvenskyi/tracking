#include <hogwidget.h>

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>

#include <QPainter>

HogWidget::HogWidget(QWidget *parent)
    : VideoWidget(parent)
    , cellCount_{ 0, 0 }
    , channelsPerCell_(0)
    , cellSize_(8)
{
    auto fillorientations = [this](QPointF *orientations, float range, int bins)
    {
        float step = range / (float)bins;
        for (int i = 0; i < bins; ++i)
        {
            float angle = (float)i * step;
            orientations[i] = QPointF(cosf(angle), sinf(angle)) * (float)cellSize_;
        }
    };
    fillorientations(orientations9_, M_PI, 9);
    fillorientations(orientations18_, 2.0f * M_PI, 18);
}

HogWidget::~HogWidget()
{}

void HogWidget::setHog(const QVector<float> &hog)
{
    QPainter painter(&frame_);
    painter.setPen(QPen(QColor(255, 255, 255)));
    painter.fillRect(frame_.rect(), QColor(0, 0, 0));
    const QPointF *orientations = bins_ == 9 ? orientations9_ : orientations18_;
    for (int y = 0; y < cellCount_[1]; ++y)
    {
        for (int x = 0; x < cellCount_[0]; ++x)
        {
            QPointF blockCenter = QPointF((float)x + 0.5f, (float)y + 0.5f) * (float)cellSize_;
            for (int b = 0; b < bins_; ++b)
            {
                float length = hog[(x + y * cellCount_[0]) * channelsPerCell_ + channelLeft_ + b];
                if (length > 1e-3f)
                {
                    painter.drawLine(blockCenter, blockCenter + orientations[b] * length);
                }
            }
        }
    }
    update();
}

void HogWidget::setUp(int cellsX, int cellsY, int channelsPerCell, int channelLeft, int bins)
{
    if (bins != 9 && bins != 18)
    {
        throw std::runtime_error("Only 9- or 18-bin histograms can be visualized");
    }
    cellCount_[0] = cellsX;
    cellCount_[1] = cellsY;
    channelsPerCell_ = channelsPerCell;
    channelLeft_ = channelLeft;
    bins_ = bins;
    frame_ = QImage(cellCount_[0] * cellSize_, cellCount_[1] * cellSize_, QImage::Format_RGB888);
    update();
}
