QT += core gui widgets
TARGET = HogApp
TEMPLATE = app

include($$PWD/../tracking.pri)
include($$PWD/../opencl.pri)
include($$PWD/../opencv.pri)

INCLUDEPATH += $$OCL_INCLUDE_DIR
LIBS += $$OCL_LIB

INCLUDEPATH += $$OCV_ROOT_DIR/include
LIBS += $$addLibsOcv(core highgui imgproc features2d calib3d)

DEPENDENCIES = VideoProcessors ImgProc VideoWidgets VideoGui
INCLUDEPATH += $$addIncludes($$DEPENDENCIES)
LIBS += $$addLibs($$DEPENDENCIES)
PRE_TARGETDEPS += $$addTargetDeps($$DEPENDENCIES)

SOURCES += \
    main.cpp \
    hogmainwin.cpp \
    hogprocessor.cpp

HEADERS += \
    hogmainwin.h \
    hogprocessor.h

FORMS += \
    hogmainwin.ui

copyFilesToDestDir($${SRC_DIR}/ImgProc/*.cl)
