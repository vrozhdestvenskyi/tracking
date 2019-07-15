#define LAB_WG_SZ 16

__kernel void rgb2lab(
    __global const uchar* restrict srcRgb,
    __global uchar* restrict dstLab,
    const int iterCnt)
{
    const float3 rgb2xyzLin[3] = {
        (float3)(0.433891f, 0.376235f, 0.189906f),
        (float3)(0.2126f, 0.7152f, 0.0722f),
        (float3)(0.0177254f, 0.109475f, 0.872955f) };

    const int iterStep = mul24((int)LAB_WG_SZ, (int)get_global_size(0));
    int idGlob = mad24(get_global_id(1), get_global_size(0), get_global_id(0));

    for (int iter = 0; iter < iterCnt; ++iter, idGlob += iterStep)
    {
        float3 v = convert_float3(vload3(idGlob, srcRgb)) * 0.003922f;
        v = select(v * 0.077399f, half_powr(mad(v, 0.947867f, 0.052133f), 2.4f), v > 0.04045f);
        v = (float3)(dot(rgb2xyzLin[0], v), dot(rgb2xyzLin[1], v), dot(rgb2xyzLin[2], v));
        v = select(mad(v, 7.787f, 0.137931f), cbrt(v), v > 0.008856f);
        v = (float3)(
            mad(v.s1, 295.8f, -40.8f),
            mad(v.s0 - v.s1, 500.0f, 128.0f),
            mad(v.s1 - v.s2, 200.0f, 128.0f));
        vstore3(convert_uchar3_sat(v + 0.5f), idGlob, dstLab);
    }
}

__kernel void lab2rgb(
    __global const uchar* restrict srcLab,
    __global uchar* restrict dstRgb,
    const int iterCnt)
{
    const float3 xyz2rgbLin[3] = {
        (float3)(3.080093f, -1.537200f, -0.542891f),
        (float3)(-0.920910f, 1.875800f, 0.045186f),
        (float3)(0.052941f, -0.204000f, 1.150893f) };

    const int iterStep = mul24((int)LAB_WG_SZ, (int)get_global_size(0));
    int idGlob = mad24(get_global_id(1), get_global_size(0), get_global_id(0));

    for (int iter = 0; iter < iterCnt; ++iter, idGlob += iterStep)
    {
        float3 v = convert_float3(vload3(idGlob, srcLab));
        float y = mad(v.s0, 0.003381f, 0.137931f);
        v = (float3)(mad(v.s1, 0.002f, y - 0.256f), y, mad(v.s2, -0.005f, y + 0.64f));
        float3 cube = v * v * v;
        v = select(mad(v, 0.128419f, -0.017712f), cube, cube > 0.008856f);
        v = (float3)(dot(xyz2rgbLin[0], v), dot(xyz2rgbLin[1], v), dot(xyz2rgbLin[2], v));
        v = select(v * 12.92f, mad(half_powr(v, 0.416667f), 1.055f, -0.055f), v > 0.0031308f);
        vstore3(convert_uchar3_sat(mad(v, 255.0f, 0.5f)), idGlob, dstRgb);
    }
}

