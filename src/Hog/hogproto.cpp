#include <hogproto.h>
#include <algorithm>

const float M_PI_FLOAT = (float)M_PI;

inline int clamp(int x, int xMin, int xMax)
{
    return std::max(std::min(x, xMax), xMin);
}

template <typename T>
inline float getPixel(const T *image, const int size[2], int x, int y)
{
    x = clamp(x, 0, size[0] - 1);
    y = clamp(y, 0, size[1] - 1);
    return (float)image[x + y * size[0]];
}

inline void setPixel(
    float *image, float value,
    const int size[2], int channels,
    int x, int y, int channel)
{
    int xClamped = clamp(x, 0, size[0] - 1);
    int yClamped = clamp(y, 0, size[1] - 1);
    int pixelIndex = (xClamped + yClamped * size[0]) * channels + channel;
    image[pixelIndex] = (x == xClamped && y == yClamped) ? value : image[pixelIndex];
}

inline void calculateBinWeights(
    float bin, int binCount,
    int interpBins[2], float interpWeights[2])
{
    interpBins[0] = (int)bin;
    interpBins[1] = interpBins[0] + 1;
    interpWeights[1] = bin - (float)interpBins[0];
    interpWeights[0] = 1.0f - interpWeights[1];
    interpBins[0] %= binCount;
    interpBins[1] %= binCount;
}

HogProto::~HogProto()
{
    release();
}

void HogProto::initialize(const HogSettings &settings)
{
    release();
    settings_ = settings;
    int cellCount = settings.cellCount_[0] * settings.cellCount_[1];
    cellSquaredNorms_ = new float [cellCount];
    std::fill(cellSquaredNorms_, cellSquaredNorms_ + cellCount, 0.0f);
    int blockCount = cellCount * 4;
    blockInverseNorms_ = new float [blockCount];
    std::fill(blockInverseNorms_, blockInverseNorms_ + blockCount, 0.0f);
    int cellDescriptorLength = cellCount * settings.channelsPerCell();
    cellDescriptor_ = new float [cellDescriptorLength];
    std::fill(cellDescriptor_, cellDescriptor_ + cellDescriptorLength, 0.0f);
    int featureDescriptorLength = cellCount * settings.channelsPerFeature();
    featureDescriptor_ = new float [featureDescriptorLength];
    std::fill(featureDescriptor_, featureDescriptor_ + featureDescriptorLength, 0.0f);
    int weightsCount = 2 * settings.cellSize_;
    cellInterpWeights_ = new float [weightsCount];
    std::fill(cellInterpWeights_, cellInterpWeights_ + weightsCount, 0.0f);
    calculateCellInterpolationWeights();
}

void HogProto::release()
{
    if (cellSquaredNorms_)
    {
        delete [] cellSquaredNorms_;
        cellSquaredNorms_ = nullptr;
    }
    if (blockInverseNorms_)
    {
        delete [] blockInverseNorms_;
        blockInverseNorms_ = nullptr;
    }
    if (cellDescriptor_)
    {
        delete [] cellDescriptor_;
        cellDescriptor_ = nullptr;
    }
    if (featureDescriptor_)
    {
        delete [] featureDescriptor_;
        featureDescriptor_ = nullptr;
    }
    if (cellInterpWeights_)
    {
        delete [] cellInterpWeights_;
        cellInterpWeights_ = nullptr;
    }
}

void HogProto::calculate(const uchar *image)
{
    calculateCellDescriptor(image);
    calculateInsensitiveNorms();
    applyNormalization();
}

void HogProto::calculateCellInterpolationWeights()
{
    for (int i = 0; i < settings_.cellSize_; ++i)
    {
        float distance = (float)i + 0.5f;
        float weight = ((float)settings_.cellSize_ - distance) / (float)settings_.cellSize_;
        cellInterpWeights_[settings_.cellSize_ - i - 1] = weight;
        cellInterpWeights_[settings_.cellSize_ + i] = weight;
    }
}

void HogProto::calculateCellDescriptor(const uchar *image)
{
    int cellSize = settings_.cellSize_;
    int channelsPerCell = settings_.channelsPerCell();
    int sensitiveBinCount = settings_.sensitiveBinCount();
    int insensitiveBinCount = settings_.insensitiveBinCount_;
    int cellCount[2] = { settings_.cellCount_[0], settings_.cellCount_[1] };
    int imageSize[2] = { cellCount[0] * cellSize, cellCount[1] * cellSize };
    int cellDescriptorLength = cellCount[0] * cellCount[1] * channelsPerCell;
    std::fill(cellDescriptor_, cellDescriptor_ + cellDescriptorLength, 0.0f);

    for (int cellY = 0; cellY < cellCount[1]; ++cellY)
    {
        int cellCenterPixelY = cellY * cellSize - cellSize / 2;
        for (int cellX = 0; cellX < cellCount[0]; ++cellX)
        {
            int cellCenterPixelX = cellX * cellSize - cellSize / 2;
            for (int cellNeighborY = 0; cellNeighborY < 2 * cellSize; ++cellNeighborY)
            {
                int pixelY = cellCenterPixelY + cellNeighborY;
                for (int cellNeighborX = 0; cellNeighborX < 2 * cellSize; ++cellNeighborX)
                {
                    int pixelX = cellCenterPixelX + cellNeighborX;

                    float gradientX =
                        getPixel<uchar>(image, imageSize, pixelX + 1, pixelY) -
                        getPixel<uchar>(image, imageSize, pixelX - 1, pixelY);
                    float gradientY =
                        getPixel<uchar>(image, imageSize, pixelX, pixelY + 1) -
                        getPixel<uchar>(image, imageSize, pixelX, pixelY - 1);
                    float magnitude = sqrtf(gradientX * gradientX + gradientY * gradientY);

                    float angle = atan2f(gradientY, gradientX);
                    angle += angle < 0.0f ? 2.0f * M_PI_FLOAT : 0.0f;
                    float bin = (float)sensitiveBinCount * angle * 0.5f / M_PI_FLOAT;

                    int interpBins[2] = { 0, 0 };
                    float interpBinWeights[2] = { 0.0f, 0.0f };
                    calculateBinWeights(bin, sensitiveBinCount, interpBins, interpBinWeights);

                    for (int i = 0; i < 2; ++i)
                    {
                        int channel = (cellX + cellY * cellCount[0]) * channelsPerCell + interpBins[i];
                        cellDescriptor_[channel] += magnitude * interpBinWeights[i] *
                            cellInterpWeights_[cellNeighborX] * cellInterpWeights_[cellNeighborY];
                    }
                }
            }
        }
    }

    int cellCountTotal = cellCount[0] * cellCount[1];
    for (int c = 0; c < cellCountTotal; ++c)
    {
        int cellShift = c * channelsPerCell;
        for (int b = 0; b < insensitiveBinCount; ++b)
        {
            cellDescriptor_[cellShift + sensitiveBinCount + b] =
                cellDescriptor_[cellShift + b] +
                cellDescriptor_[cellShift + insensitiveBinCount + b];
        }
    }
}

void HogProto::calculateInsensitiveNorms()
{
    int channelsPerCell = settings_.channelsPerCell();
    int sensitiveBinCount = settings_.sensitiveBinCount();
    int insensitiveBinCount = settings_.insensitiveBinCount_;
    int cellCount[2] = { settings_.cellCount_[0], settings_.cellCount_[1] };
    int cellCountTotal = cellCount[0] * cellCount[1];
    const float *cellDescriptor = cellDescriptor_ + sensitiveBinCount;
    std::fill(cellSquaredNorms_, cellSquaredNorms_ + cellCountTotal, 0.0f);
    std::fill(blockInverseNorms_, blockInverseNorms_ + cellCount[0] * cellCount[1] * 4, 0.0f);

    for (int c = 0; c < cellCountTotal; ++c)
    {
        for (int b = 0; b < insensitiveBinCount; ++b)
        {
            float magnitude = cellDescriptor[c * channelsPerCell + b];
            cellSquaredNorms_[c] += magnitude * magnitude;
        }
    }
    for (int y = -1; y <= cellCount[1]; ++y)
    {
        for (int x = -1; x <= cellCount[0]; ++x)
        {
            float inverseNorm = 1.0f / sqrtf(
                getPixel<float>(cellSquaredNorms_, cellCount, x, y) +
                getPixel<float>(cellSquaredNorms_, cellCount, x + 1, y) +
                getPixel<float>(cellSquaredNorms_, cellCount, x, y + 1) +
                getPixel<float>(cellSquaredNorms_, cellCount, x + 1, y + 1) +
                1e-8f);
            setPixel(blockInverseNorms_, inverseNorm, cellCount, 4, x, y, 0);
            setPixel(blockInverseNorms_, inverseNorm, cellCount, 4, x + 1, y, 1);
            setPixel(blockInverseNorms_, inverseNorm, cellCount, 4, x, y + 1, 2);
            setPixel(blockInverseNorms_, inverseNorm, cellCount, 4, x + 1, y + 1, 3);
        }
    }
}

void HogProto::applyNormalization()
{
    int sensitiveBinCount = settings_.sensitiveBinCount();
    int insensitiveBinCount = settings_.insensitiveBinCount_;
    int channelsPerCell = settings_.channelsPerCell();
    int channelsPerFeature = settings_.channelsPerFeature();
    int cellCountTotal = settings_.cellCount_[0] * settings_.cellCount_[1];
    float truncation = settings_.truncation_;
    std::fill(featureDescriptor_, featureDescriptor_ + cellCountTotal * channelsPerFeature, 0.0f);

    for (int c = 0; c < cellCountTotal; ++c)
    {
        for (int b = 0; b < channelsPerCell; ++b)
        {
            float unnormalized = cellDescriptor_[c * channelsPerCell + b];
            float normalized = 0.0f;
            for (int i = 0; i < 4; ++i)
            {
                normalized += 0.5f * fminf(unnormalized * blockInverseNorms_[c * 4 + i], truncation);
            }
            featureDescriptor_[c * channelsPerFeature + b] = normalized;
        }
        for (int i = 0; i < 4; ++i)
        {
            float normalization = blockInverseNorms_[c * 4 + i];
            float normalized = 0.0f;
            for (int b = 0; b < insensitiveBinCount; ++b)
            {
                normalized +=  0.2357f * fminf(
                    normalization * cellDescriptor_[c * channelsPerCell + sensitiveBinCount + b],
                    truncation);
            }
            featureDescriptor_[c * channelsPerFeature + channelsPerCell + i] = normalized;
        }
    }
}

