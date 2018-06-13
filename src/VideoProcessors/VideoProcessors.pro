TARGET = VideoProcessors
TEMPLATE = lib
CONFIG += staticlib

include($$PWD/../tracking.pri)

# OpenCL
INCLUDEPATH += "C:\Intel\OpenCL\sdk\include"

SOURCES += \
    videoprocessor.cpp \
    oclprocessor.cpp

HEADERS += \
    videoprocessor.h \
    oclprocessor.h
