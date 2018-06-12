TARGET = HogAppSln
TEMPLATE = subdirs

include($$PWD/../../src/tracking.pri)

SUBDIRS += \
    HogApp \
    Hog \
    HogPiotr \
    VideoProcessors \
    VideoWidgets \
    VideoGui

HogApp.subdir = $$SRC_DIR/HogApp
Hog.subdir = $$SRC_DIR/Hog
HogPiotr.subdir = $$SRC_DIR/HogPiotr
VideoProcessors.subdir = $$SRC_DIR/VideoProcessors
VideoWidgets.subdir = $$SRC_DIR/VideoWidgets
VideoGui.subdir = $$SRC_DIR/VideoGui

HogApp.depends = \
    Hog \
    HogPiotr \
    VideoProcessors \
    VideoWidgets \
    VideoGui
