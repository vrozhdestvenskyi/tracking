QT -= gui
TARGET = Hog
TEMPLATE = lib
CONFIG += staticlib

include($$PWD/../tracking.pri)

SOURCES += \
    hog.cpp

HEADERS += \
    hog.h
