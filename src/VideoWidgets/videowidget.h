#ifndef VIDEOWIDGET_H
#define VIDEOWIDGET_H

#include <QWidget>

class VideoWidget : public QWidget
{
    Q_OBJECT

public:
    VideoWidget(QWidget *parent = nullptr);
    virtual ~VideoWidget() override;

public slots:
    void setFrame(const QImage &frame);

protected:
    virtual void paintEvent(QPaintEvent *event) override;

protected:
    QImage frame_;
};

#endif // VIDEOWIDGET_H
