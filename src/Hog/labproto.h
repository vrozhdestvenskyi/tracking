#ifndef LABPROTO_H
#define LABPROTO_H

using uchar = unsigned char;

void rgb2lab(const uchar *srcRgb, int sz, uchar *dstLab);
void lab2rgb(const uchar *srcLab, int sz, uchar *dstRgb);

#endif // LABPROTO_H

