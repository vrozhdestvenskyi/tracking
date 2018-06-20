QT += widgets

TARGET = VideoGui
TEMPLATE = lib
CONFIG += staticlib

include($$PWD/../tracking.pri)
include($$PWD/../opencl.pri)

INCLUDEPATH += $$OCL_INCLUDE_DIR
INCLUDEPATH += $$addIncludes(VideoProcessors)

SOURCES += \
    videocapturebase.cpp

HEADERS += \
    videocapturebase.h
