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
            mad(v.s1, 295.8f, 40.8f),
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
    // TODO xyzWeights may be encapsulated into xyz2rgbLin
}

