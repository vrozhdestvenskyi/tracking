#ifndef HOGWIDGET_H
#define HOGWIDGET_H

#include <videowidget.h>

class HogWidget : public VideoWidget
{
    Q_OBJECT

public:
    HogWidget(QWidget *parent = nullptr);
    virtual ~HogWidget() override;

public slots:
    /// Note: it is assumed that all the values are normalized to [0, 0.5)
    void setHog(const QVector<float> &hog);
    void setUp(int cellsX, int cellsY, int channelsPerCell, int channelLeft, int bins);

protected:
    int cellCount_[2] = { 0, 0 };
    int channelsPerCell_ = 0;
    int cellSize_ = 0;
    int channelLeft_ = 0;
    int bins_ = 0;
    QPointF orientations9_[9];
    QPointF orientations18_[18];
};

#endif // HOGWIDGET_H
