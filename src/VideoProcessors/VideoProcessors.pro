TARGET = VideoProcessors
TEMPLATE = lib
CONFIG += staticlib

include($$PWD/../tracking.pri)

SOURCES += \
    videoprocessor.cpp

HEADERS += \
    videoprocessor.h
