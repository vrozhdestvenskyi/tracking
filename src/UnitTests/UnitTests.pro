QT -= gui
QT += core widgets
TARGET = UnitTests
TEMPLATE = app

include($$PWD/../tracking.pri)
include($$PWD/../opencl.pri)
include($$PWD/../opencv.pri)
include($$PWD/../googletest.pri)

INCLUDEPATH += \
    $$OCL_INCLUDE_DIR \
    $$OCV_ROOT_DIR/include \
    $$GTEST_INCLUDE_DIR
LIBS += \
    $$OCL_LIB \
    $$addLibsOcv(core highgui imgproc features2d calib3d) \
    $$addLibsGtest(gtest)

# For Piotr's HOG implementation
CONFIG += mmx sse sse2
QMAKE_FLAGS += -msse4.1 -mssse3 -msse3 -msse2 -msse
QMAKE_CXXFLAGS += -msse4.1 -mssse3 -msse3 -msse2 -msse

DEPENDENCIES = ImgProc HogPiotr VideoProcessors
INCLUDEPATH += $$addIncludes($$DEPENDENCIES)
LIBS += $$addLibs($$DEPENDENCIES)
PRE_TARGETDEPS += $$addTargetDeps($$DEPENDENCIES)

SOURCES += \
    colorconversionstest.cpp \
    ffttest.cpp \
    hogtest.cpp \
    main.cpp

HEADERS += \
    testhelpers.h

copyFilesToDestDir($${SRC_DIR}/ImgProc/*.cl)
