#include <QColor>
#include <QPainter>
#include <videowidget.h>

VideoWidget::VideoWidget(QWidget *parent)
    : QWidget(parent)
    , frame_(8, 8, QImage::Format_RGB888)
{
    frame_.fill(QColor(0, 0, 0));
}

VideoWidget::~VideoWidget()
{}

void VideoWidget::setFrame(const QImage &frame)
{
    frame_ = frame.convertToFormat(QImage::Format_RGB888);
    update();
}

void VideoWidget::paintEvent(QPaintEvent *event)
{
    QWidget::paintEvent(event);

    QRect frameRect = rect();
    if (frame_.width() * height() > frame_.height() * width())
    {
        frameRect.setHeight(frame_.height() * width() / frame_.width());
        frameRect.translate(0, (height() - frameRect.height()) / 2);
    }
    if (frame_.width() * height() < frame_.height() * width())
    {
        frameRect.setWidth(frame_.width() * height() / frame_.height());
        frameRect.translate((width() - frameRect.width()) / 2, 0);
    }

    QPainter painter(this);
    painter.fillRect(rect(), QColor(0, 0, 0));
    painter.setViewport(frameRect);
    painter.setWindow(0, 0, frame_.width(), frame_.height());
    painter.drawImage(painter.window(), frame_);
}
