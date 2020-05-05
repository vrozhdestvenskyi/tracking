QT -= gui
TARGET = ImgProc
TEMPLATE = lib
CONFIG += staticlib

include($$PWD/../tracking.pri)
include($$PWD/../opencl.pri)

INCLUDEPATH += $$OCL_INCLUDE_DIR

SOURCES += \
    colorconversions.cpp \
    colorconversionsproto.cpp \
    fftproto.cpp \
    hogproto.cpp \
    hog.cpp \
    rangedkernel.cpp

HEADERS += \
    colorconversions.h \
    colorconversionsproto.h \
    fftproto.h \
    hogproto.h \
    hog.h \
    rangedkernel.h

DISTFILES += \
    colorconversions.cl \
    fft.cl \
    hog.cl
