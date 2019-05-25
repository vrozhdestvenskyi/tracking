#include <labproto.h>
#include <cmath>

inline uchar clamp8bit(const float v)
{
    return static_cast<uchar>(fmaxf(0.0f, fminf(255.0f, v)) + 0.5f);
}

void rgb2lab(const uchar *srcRgb, int sz, uchar *dstLab)
{
    const float rgb2xyzLin[3][3] = {
        { 0.433891f, 0.376235f, 0.189906f },
        { 0.2126f, 0.7152f, 0.0722f },
        { 0.0177254f, 0.109475f, 0.872955f } };
    float rgb[3] = { 0.0f, 0.0f, 0.0f };
    float xyz[3] = { 0.0f, 0.0f, 0.0f };
    for (int pixId = 0; pixId < sz; ++pixId)
    {
        for (int i = 0; i < 3; ++i)
        {
            rgb[i] = srcRgb[pixId * 3 + i] / 255.0f;
            rgb[i] = rgb[i] > 0.04045f
                ? powf((rgb[i] + 0.055f) / 1.055f, 2.4f)
                : (rgb[i] / 12.92f);
        }
        for (int i = 0; i < 3; ++i)
        {
            xyz[i] = 0.0f;
            for (int j = 0; j < 3; ++j)
            {
                xyz[i] += rgb2xyzLin[i][j] * rgb[j];
            }
        }
//        xyz[0] /= 95.0489f;
//        xyz[1] /= 100.0f;
//        xyz[2] /= 108.8840f;
        for (int i = 0; i < 3; ++i)
        {
            xyz[i] = xyz[i] > 0.008856f
                ? cbrtf(xyz[i])
                : (7.787f * xyz[i] + 16.0f / 116.0f);
        }
        dstLab[pixId * 3] = clamp8bit((116.0f * xyz[1] - 16.0f) * 2.55f);
        dstLab[pixId * 3 + 1] = clamp8bit(500.0f * (xyz[0] - xyz[1]) + 128.0f);
        dstLab[pixId * 3 + 2] = clamp8bit(200.0f * (xyz[1] - xyz[2]) + 128.0f);
    }
}

void lab2rgb(const uchar *srcLab, int sz, uchar *dstRgb)
{
    const float xyzWeights[3] = { 0.95047f, 1.0f, 1.08883f };
    const float xyz2rgbLin[3][3] = {
        { 3.2406f, -1.5372f, -0.4986f },
        { -0.9689f, 1.8758f, 0.0415f },
        { 0.0557f, -0.2040f, 1.0570f } };
    float xyz[3] = { 0.0f, 0.0f, 0.0f };
    for (int pixId = 0; pixId < sz; ++pixId)
    {
        xyz[1] = (srcLab[pixId * 3] / 2.55f + 16.0f) / 116.0f;
        xyz[0] = (srcLab[pixId * 3 + 1] - 128.0f) / 500.0f + xyz[1];
        xyz[2] = xyz[1] - (srcLab[pixId * 3 + 2] - 128.0f) / 200.0f;
        for (int i = 0; i < 3; ++i)
        {
            float cube = xyz[i] * xyz[i] * xyz[i];
            xyz[i] = xyzWeights[i] * (cube > 0.008856f
                ? cube
                : ((xyz[i] - 16.0f / 116.0f) / 7.787f));
        }
        for (int i = 0; i < 3; ++i)
        {
            float rgb = 0.0f;
            for (int j = 0; j < 3; ++j)
            {
                rgb += xyz2rgbLin[i][j] * xyz[j];
            }
            rgb = rgb > 0.0031308f
                ? (1.055f * powf(rgb, 1.0f / 2.4f) - 0.055f)
                : (12.92f * rgb);
            dstRgb[pixId * 3 + i] = clamp8bit(rgb * 255.0f);
        }
    }
}

