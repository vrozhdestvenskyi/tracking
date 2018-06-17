QT -= gui
TARGET = Hog
TEMPLATE = lib
CONFIG += staticlib

include($$PWD/../tracking.pri)
include($$PWD/../opencl.pri)

INCLUDEPATH += $$OCL_INCLUDE_DIR

SOURCES += \
    hogproto.cpp \
    hog.cpp

HEADERS += \
    hogproto.h \
    hog.h

DISTFILES += \
    hog.cl
