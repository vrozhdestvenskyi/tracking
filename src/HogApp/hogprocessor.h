#ifndef HOGPROCESSOR_H
#define HOGPROCESSOR_H

#include <memory>
#include <QElapsedTimer>
#include <hog.h>
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
    void calculateHogOcl();
    void calculateHogPiotr();
    void compareDescriptors(const float *desc) const;
    void compareDescriptorsOcl(const float *mappedDescriptor) const;

    std::shared_ptr<cv::Mat_<uchar> > ocvImageGray_ = nullptr;
    std::shared_ptr<cv::Mat_<float> > ocvImageGrayFloat_ = nullptr;
    float *hogPiotr_ = nullptr;
    cl_mem oclImage_ = NULL;
    Derivs derivs_;
    HogProto hogProto_;
    CellHog hog_;
    CellNorm cellNorm_;
    CellNormSumX cellNormSumX_;
    InvBlockNorm invBlockNorm_;
    BlockHog blockHog_;
    QElapsedTimer timer_;
    quint64 msSum_ = 0;
};

#endif // HOGPROCESSOR_H
