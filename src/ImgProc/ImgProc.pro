QT -= gui
TARGET = ImgProc
TEMPLATE = lib
CONFIG += staticlib

include($$PWD/../tracking.pri)
include($$PWD/../opencl.pri)

INCLUDEPATH += $$OCL_INCLUDE_DIR

SOURCES += \
    hogproto.cpp \
    hog.cpp \
    labproto.cpp \
    lab.cpp \
    rangedkernel.cpp

HEADERS += \
    hogproto.h \
    hog.h \
    labproto.h \
    lab.h \
    rangedkernel.h

DISTFILES += \
    hog.cl \
    lab.cl
