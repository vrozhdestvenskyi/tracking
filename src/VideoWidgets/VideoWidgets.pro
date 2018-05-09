QT += widgets
QT -= gui

TARGET = VideoWidgets
TEMPLATE = lib
CONFIG += staticlib

include($$PWD/../tracking.pri)

SOURCES += \
        videowidget.cpp \
    hogwidget.cpp

HEADERS += \
        videowidget.h \
    hogwidget.h
unix {
    target.path = /usr/lib
    INSTALLS += target
}
