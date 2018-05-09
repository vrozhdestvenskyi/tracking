QT += core gui widgets
TARGET = VideoCaptureApp
TEMPLATE = app

include($$PWD/../tracking.pri)
include($$PWD/../opencv.pri)

INCLUDEPATH += \
    $$OCV_ROOT_DIR/include

LIBS += \
    -L$$OCV_BIN_DIR \
    -l$$ocvLibName(core) \
    -l$$ocvLibName(highgui) \
    -l$$ocvLibName(imgproc) \
    -l$$ocvLibName(features2d) \
    -l$$ocvLibName(calib3d)

INCLUDEPATH += \
    $$SRC_DIR\VideoProcessors \
    $$SRC_DIR\VideoWidgets \
    $$SRC_DIR\VideoGui

LIBS += \
    -L$$projectBinDir(VideoProcessors) -lVideoProcessors \
    -L$$projectBinDir(VideoWidgets) -lVideoWidgets \
    -L$$projectBinDir(VideoGui) -lVideoGui

SOURCES += \
    main.cpp \
    videocapturemainwin.cpp

HEADERS += \
    videocapturemainwin.h

FORMS += \
    videocapturemainwin.ui
