QT -= gui
TARGET = Hog
TEMPLATE = lib
CONFIG += staticlib

include($$PWD/../tracking.pri)
include($$PWD/../opencv.pri)

INCLUDEPATH += \
    $$OCV_ROOT_DIR/include

SOURCES += \
    hog.cpp

HEADERS += \
    hog.h \
    piotr_fhog.hpp \
    piotr_gradientMex.h \
    piotr_sse.hpp \
    piotr_wrappers.hpp

unix {
    target.path = /usr/lib
    INSTALLS += target
}
