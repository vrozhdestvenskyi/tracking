QT -= gui
TARGET = Hog
TEMPLATE = lib
CONFIG += staticlib

include($$PWD/../tracking.pri)
include($$PWD/../opencl.pri)

INCLUDEPATH += $$OCL_INCLUDE_DIR

SOURCES += \
    hogproto.cpp \
    hog.cpp \
    labproto.cpp

HEADERS += \
    hogproto.h \
    hog.h \
    labproto.h

DISTFILES += \
    hog.cl
