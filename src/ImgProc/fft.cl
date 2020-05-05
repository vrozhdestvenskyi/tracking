#define DFT_2(c0, c1) \
{                     \
    float2 v0;        \
    v0 = c0;          \
    c0 = v0 + c1;     \
    c1 = v0 - c1;     \
}

#define SQRT3DIV2 0.86602540378443f

#define DFT_3(c0, c1, c2)                          \
{                                                  \
    float2 v0 = c1 + c2;                           \
    float2 v1 = c1 - c2;                           \
    c1.x = c0.x - 0.5f * v0.x + v1.y * SQRT3DIV2;  \
    c1.y = c0.y - 0.5f * v0.y - v1.x * SQRT3DIV2;  \
    c2.x = c0.x - 0.5f * v0.x - v1.y * SQRT3DIV2;  \
    c2.y = c0.y - 0.5f * v0.y + v1.x * SQRT3DIV2;  \
    c0 = c0 + v0;                                  \
}

#define DFT_4(c0, c1, c2, c3) \
{                             \
    float2 v0, v1, v2, v3;    \
    v0 = c0 + c2;             \
    v1 = c1 + c3;             \
    v2 = c0 - c2;             \
    v3.x = c1.y - c3.y;       \
    v3.y = c3.x - c1.x;       \
    c0 = v0 + v1;             \
    c2 = v0 - v1;             \
    c1 = v2 + v3;             \
    c3 = v2 - v3;             \
}

#define W5_A 0.30901699437494f
#define W5_B 0.95105651629515f
#define W5_C 0.80901699437494f
#define W5_D 0.58778525229247f

#define DFT_5(c0, c1, c2, c3, c4)               \
{                                               \
    float2 v0, v1, v2, v3, v4;                  \
    v0 = c0;                                    \
    v1 = W5_A * (c1 + c4) - W5_C * (c2 + c3);   \
    v2 = W5_C * (c1 + c4) - W5_A * (c2 + c3);   \
    v3 = W5_D * (c1 - c4) - W5_B * (c2 - c3);   \
    v4 = W5_B * (c1 - c4) + W5_D * (c2 - c3);   \
    c0 = v0 + c1 + c2 + c3 + c4;                \
    c1 = v0 + v1 + (float2)(v4.y, -v4.x);       \
    c2 = v0 - v2 + (float2)(v3.y, -v3.x);       \
    c3 = v0 - v2 + (float2)(-v3.y, v3.x);       \
    c4 = v0 + v1 + (float2)(-v4.y, v4.x);       \
}

#define W7_A 0.62348980185873f
#define W7_B 0.78183148246802f
#define W7_C 0.22252093395631f
#define W7_D 0.97492791218182f
#define W7_E 0.90096886790241f
#define W7_F 0.43388373911755f

#define DFT_7(c0, c1, c2, c3, c4, c5, c6)                           \
{                                                                   \
    float2 v0, v1, v2, v3, v4, v5, v6;                              \
    v0 = c0;                                                        \
    v1 = W7_A * (c1 + c6) - W7_C * (c2 + c5) - W7_E * (c3 + c4);    \
    v2 = W7_C * (c1 + c6) + W7_E * (c2 + c5) - W7_A * (c3 + c4);    \
    v3 = W7_E * (c1 + c6) - W7_A * (c2 + c5) + W7_C * (c3 + c4);    \
    v4 = W7_B * (c1 - c6) + W7_D * (c2 - c5) + W7_F * (c3 - c4);    \
    v5 = W7_D * (c1 - c6) - W7_F * (c2 - c5) - W7_B * (c3 - c4);    \
    v6 = W7_F * (c1 - c6) - W7_B * (c2 - c5) + W7_D * (c3 - c4);    \
    c0 = v0 + c1 + c2 + c3 + c4 + c5 + c6;                          \
    c1 = v0 + v1 + (float2)(v4.y, -v4.x);                           \
    c2 = v0 - v2 + (float2)(v5.y, -v5.x);                           \
    c3 = v0 - v3 + (float2)(v6.y, -v6.x);                           \
    c4 = v0 - v3 + (float2)(-v6.y, v6.x);                           \
    c5 = v0 - v2 + (float2)(-v5.y, v5.x);                           \
    c6 = v0 + v1 + (float2)(-v4.y, v4.x);                           \
}

#define M_2PI_F 6.283185482025146484375f

#define TWIDDLE_FACTOR_MULTIPLICATION(phi, input)                 \
{                                                                 \
    float2 w, tmp;                                                \
    w.x = cos(phi);                                               \
    w.y = sin(phi);                                               \
    tmp.x = (w.x * input.x) - (w.y * input.y);                    \
    tmp.y = (w.x * input.y) + (w.y * input.x);                    \
    input = tmp;                                                  \
}

kernel void fft()
{
    /* Init Nx to 1 */
    uint Nx = 1;

    /* Scan each radix stage */
    for(uint s = 0; s < n_stages; ++s)
    {
         /* Get radix order of stage s */
         uint Ny = radix[s];

         /* Body for computing twiddle factor multiplication and radix computation */
         ...

         /* Update Nx */
         Nx *= Ny;
    }
}

