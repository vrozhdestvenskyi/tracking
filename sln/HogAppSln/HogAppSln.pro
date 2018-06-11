TARGET = HogAppSln
TEMPLATE = subdirs

include($$PWD/../../src/tracking.pri)

SUBDIRS += \
    HogApp \
    VideoProcessors \
    Hog \
    HogPiotr \
    VideoWidgets \
    VideoGui

HogApp.subdir = $$SRC_DIR/HogApp
VideoProcessors.subdir = $$SRC_DIR/VideoProcessors
Hog.subdir = $$SRC_DIR/Hog
HogPiotr.subdir = $$SRC_DIR/HogPiotr
VideoWidgets.subdir = $$SRC_DIR/VideoWidgets
VideoGui.subdir = $$SRC_DIR/VideoGui

VideoProcessors.depends = \
    Hog \
    HogPiotr

HogApp.depends = \
    VideoProcessors \
    VideoWidgets \
    VideoGui
