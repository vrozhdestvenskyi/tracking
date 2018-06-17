#ifndef HOGPROCESSOR_H
#define HOGPROCESSOR_H

#include <memory>
#include <hogproto.h>
#include <videoprocessor.h>

namespace cv
{
template <typename T>
class Mat_;
}

class HogProcessor : public VideoProcessor
{
    Q_OBJECT

public:
    HogProcessor(QObject *parent = nullptr);
    ~HogProcessor() override;

public slots:
    void processFrame() override;
    void setupProcessor(const VideoProcessor::CaptureSettings &settings) override;

signals:
    void sendHog(const QVector<float> &hog);
    void sendHogSettings(int cellsX, int cellsY, int channelsPerCell, int channelLeft, int bins);

protected:
    void release();
    void calculateHogPiotr();
    void compareDescriptors() const;

    std::shared_ptr<cv::Mat_<uchar> > ocvImageGray_ = nullptr;
    std::shared_ptr<cv::Mat_<float> > ocvImageGrayFloat_ = nullptr;
    float *hogPiotr_ = nullptr;
    HogProto hogProto_;
};

#endif // HOGPROCESSOR_H
