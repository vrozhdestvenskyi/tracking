QT -= gui
TARGET = HogPiotr
TEMPLATE = lib
CONFIG += staticlib

include($$PWD/../tracking.pri)
include($$PWD/../opencv.pri)

INCLUDEPATH += \
    $$OCV_ROOT_DIR/include

HEADERS += \
    fhog.hpp \
    gradientMex.h \
    sse.hpp \
    wrappers.hpp
