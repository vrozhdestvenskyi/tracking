#ifndef COLORCONVERSIONSPROTO_H
#define COLORCONVERSIONSPROTO_H

using uchar = unsigned char;

void rgb2lab(const uchar *srcRgb, int sz, uchar *dstLab);
void lab2rgb(const uchar *srcLab, int sz, uchar *dstRgb);

#endif // COLORCONVERSIONSPROTO_H

