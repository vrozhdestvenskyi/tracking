#include <hog.h>
#include <algorithm>

const float M_PI_FLOAT = (float)M_PI;

void calculateCellInterpolationWeights(HogHandle &handle)
{
    const HogSettings &settings = handle.settings_;
    for (int i = 0; i < settings.cellSize_; ++i)
    {
        float distance = (float)i + 0.5f;
        float weight = ((float)settings.cellSize_ - distance) / (float)settings.cellSize_;
        handle.cellInterpWeights_[settings.cellSize_ - i - 1] = weight;
        handle.cellInterpWeights_[settings.cellSize_ + i] = weight;
    }
}

void initializeHogHandle(const HogSettings &settings, HogHandle &handle)
{
    releaseHogHandle(handle);
    handle.settings_ = settings;
    int cellCount = settings.cellCount_[0] * settings.cellCount_[1];
    handle.cellSquaredNorms_ = new float [cellCount];
    std::fill(handle.cellSquaredNorms_, handle.cellSquaredNorms_ + cellCount, 0.0f);
    int blockCount = cellCount * 4;
    handle.blockInverseNorms_ = new float [blockCount];
    std::fill(handle.blockInverseNorms_, handle.blockInverseNorms_ + blockCount, 0.0f);
    int cellDescriptorLength = cellCount * settings.channelsPerCell();
    handle.cellDescriptor_ = new float [cellDescriptorLength];
    std::fill(handle.cellDescriptor_, handle.cellDescriptor_ + cellDescriptorLength, 0.0f);
    int featureDescriptorLength = cellCount * settings.channelsPerFeature();
    handle.featureDescriptor_ = new float [featureDescriptorLength];
    std::fill(handle.featureDescriptor_, handle.featureDescriptor_ + featureDescriptorLength, 0.0f);
    int weightsCount = 2 * settings.cellSize_;
    handle.cellInterpWeights_ = new float [weightsCount];
    std::fill(handle.cellInterpWeights_, handle.cellInterpWeights_ + weightsCount, 0.0f);
    calculateCellInterpolationWeights(handle);;
}

void releaseHogHandle(HogHandle &handle)
{
    if (handle.cellSquaredNorms_)
    {
        delete [] handle.cellSquaredNorms_;
        handle.cellSquaredNorms_ = nullptr;
    }
    if (handle.blockInverseNorms_)
    {
        delete [] handle.blockInverseNorms_;
        handle.blockInverseNorms_ = nullptr;
    }
    if (handle.cellDescriptor_)
    {
        delete [] handle.cellDescriptor_;
        handle.cellDescriptor_ = nullptr;
    }
    if (handle.featureDescriptor_)
    {
        delete [] handle.featureDescriptor_;
        handle.featureDescriptor_ = nullptr;
    }
    if (handle.cellInterpWeights_)
    {
        delete [] handle.cellInterpWeights_;
        handle.cellInterpWeights_ = nullptr;
    }
}

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

void calculateCellDescriptor(const uchar *image, HogHandle &handle)
{
    const HogSettings &settings = handle.settings_;
    const float *cellInterpWeights = handle.cellInterpWeights_;
    int cellSize = settings.cellSize_;
    int channelsPerCell = settings.channelsPerCell();
    int sensitiveBinCount = settings.sensitiveBinCount();
    int insensitiveBinCount = settings.insensitiveBinCount_;
    int cellCount[2] = { settings.cellCount_[0], settings.cellCount_[1] };
    int imageSize[2] = { cellCount[0] * cellSize, cellCount[1] * cellSize };

    float *cellDescriptor = handle.cellDescriptor_;
    int length = cellCount[0] * cellCount[1] * channelsPerCell;
    std::fill(cellDescriptor, cellDescriptor + length, 0.0f);

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
                        cellDescriptor[channel] += magnitude * interpBinWeights[i] *
                            cellInterpWeights[cellNeighborX] * cellInterpWeights[cellNeighborY];
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
            cellDescriptor[cellShift + sensitiveBinCount + b] =
                cellDescriptor[cellShift + b] +
                cellDescriptor[cellShift + insensitiveBinCount + b];
        }
    }
}

void calculateInsensitiveNorms(HogHandle &handle)
{
    const HogSettings &settings = handle.settings_;
    const float *cellDescriptor = handle.cellDescriptor_;
    int channelsPerCell = settings.channelsPerCell();
    int sensitiveBinCount = settings.sensitiveBinCount();
    int insensitiveBinCount = settings.insensitiveBinCount_;
    int cellCount[2] = { settings.cellCount_[0], settings.cellCount_[1] };
    int cellCountTotal = cellCount[0] * cellCount[1];

    float *cellSquaredNorms = handle.cellSquaredNorms_;
    std::fill(cellSquaredNorms, cellSquaredNorms + cellCountTotal, 0.0f);
    cellDescriptor += sensitiveBinCount;

    for (int c = 0; c < cellCountTotal; ++c)
    {
        for (int b = 0; b < insensitiveBinCount; ++b)
        {
            float magnitude = cellDescriptor[c * channelsPerCell + b];
            cellSquaredNorms[c] += magnitude * magnitude;
        }
    }

    float *blockInverseNorms = handle.blockInverseNorms_;
    std::fill(blockInverseNorms, blockInverseNorms + cellCount[0] * cellCount[1] * 4, 0.0f);

    for (int y = -1; y <= cellCount[1]; ++y)
    {
        for (int x = -1; x <= cellCount[0]; ++x)
        {
            float inverseNorm = 1.0f / sqrtf(
                getPixel<float>(cellSquaredNorms, cellCount, x, y) +
                getPixel<float>(cellSquaredNorms, cellCount, x + 1, y) +
                getPixel<float>(cellSquaredNorms, cellCount, x, y + 1) +
                getPixel<float>(cellSquaredNorms, cellCount, x + 1, y + 1) +
                1e-8f
            );
            setPixel(blockInverseNorms, inverseNorm, cellCount, 4, x, y, 0);
            setPixel(blockInverseNorms, inverseNorm, cellCount, 4, x + 1, y, 1);
            setPixel(blockInverseNorms, inverseNorm, cellCount, 4, x, y + 1, 2);
            setPixel(blockInverseNorms, inverseNorm, cellCount, 4, x + 1, y + 1, 3);
        }
    }
}

void applyNormalization(HogHandle &handle)
{
    const HogSettings &settings = handle.settings_;
    const float *blockInverseNorms = handle.blockInverseNorms_;
    const float *cellDescriptor = handle.cellDescriptor_;
    int sensitiveBinCount = settings.sensitiveBinCount();
    int insensitiveBinCount = settings.insensitiveBinCount_;
    int channelsPerCell = settings.channelsPerCell();
    int channelsPerFeature = settings.channelsPerFeature();
    int cellCountTotal = settings.cellCount_[0] * settings.cellCount_[1];
    float truncation = settings.truncation_;

    float *featureDescriptor = handle.featureDescriptor_;
    std::fill(featureDescriptor, featureDescriptor + cellCountTotal * channelsPerFeature, 0.0f);

    for (int c = 0; c < cellCountTotal; ++c)
    {
        for (int b = 0; b < channelsPerCell; ++b)
        {
            float unnormalized = cellDescriptor[c * channelsPerCell + b];
            float normalized = 0.0f;
            for (int i = 0; i < 4; ++i)
            {
                normalized += fminf(
                    unnormalized * blockInverseNorms[c * 4 + i], truncation
                ) * 0.5f;
            }
            featureDescriptor[c * channelsPerFeature + b] = normalized;
        }

        for (int i = 0; i < 4; ++i)
        {
            float normalization = blockInverseNorms[c * 4 + i];
            float normalized = 0.0f;
            for (int b = 0; b < insensitiveBinCount; ++b)
            {
                normalized += fminf(
                    normalization * cellDescriptor[c * channelsPerCell + sensitiveBinCount + b],
                    truncation
                ) * 0.2357f;
            }
            featureDescriptor[c * channelsPerFeature + channelsPerCell + i] = normalized;
        }
    }
}

void calculateHog(const uchar *image, HogHandle &handle)
{
    calculateCellDescriptor(image, handle);
    calculateInsensitiveNorms(handle);
    applyNormalization(handle);
}

