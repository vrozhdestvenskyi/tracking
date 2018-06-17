TARGET = VideoProcessors
TEMPLATE = lib
CONFIG += staticlib

include($$PWD/../tracking.pri)
include($$PWD/../opencl.pri)

INCLUDEPATH += $$OCL_INCLUDE_DIR

SOURCES += \
    videoprocessor.cpp \
    oclprocessor.cpp

HEADERS += \
    videoprocessor.h \
    oclprocessor.h
