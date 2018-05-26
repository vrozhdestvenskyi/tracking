TARGET = VideoProcessors
TEMPLATE = lib
CONFIG += staticlib

# For Piotr's HOG implementation
CONFIG += mmx sse sse2
QMAKE_FLAGS += -msse4.1 -mssse3 -msse3 -msse2 -msse
QMAKE_CXXFLAGS += -msse4.1 -mssse3 -msse3 -msse2 -msse

include($$PWD/../tracking.pri)
include($$PWD/../opencv.pri)

INCLUDEPATH += \
    $$OCV_ROOT_DIR/include

INCLUDEPATH += \
    $$SRC_DIR\Hog

SOURCES += \
    videoprocessor.cpp \
    hogprocessor.cpp

HEADERS += \
    videoprocessor.h \
    hogprocessor.h
