TARGET = HogAppSln
TEMPLATE = subdirs

include($$PWD/../../src/tracking.pri)

SUBDIRS += \
    HogApp \
    ImgProc \
    HogPiotr \
    VideoProcessors \
    VideoWidgets \
    VideoGui

HogApp.subdir = $$SRC_DIR/HogApp
ImgProc.subdir = $$SRC_DIR/ImgProc
HogPiotr.subdir = $$SRC_DIR/HogPiotr
VideoProcessors.subdir = $$SRC_DIR/VideoProcessors
VideoWidgets.subdir = $$SRC_DIR/VideoWidgets
VideoGui.subdir = $$SRC_DIR/VideoGui

VideoGui.depends = \
    VideoProcessors

HogApp.depends = \
    ImgProc \
    HogPiotr \
    VideoGui \
    VideoWidgets
