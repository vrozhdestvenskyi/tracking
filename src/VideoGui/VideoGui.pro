QT += widgets

TARGET = VideoGui
TEMPLATE = lib
CONFIG += staticlib

include($$PWD/../tracking.pri)

INCLUDEPATH += \
    $$SRC_DIR\VideoProcessors

SOURCES += \
    videocapturebase.cpp

HEADERS += \
    videocapturebase.h

unix {
    target.path = /usr/lib
    INSTALLS += target
}
