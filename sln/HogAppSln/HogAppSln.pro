TARGET = HogAppSln
TEMPLATE = subdirs

include($$PWD/../../src/tracking.pri)

SUBDIRS += \
    HogApp \
    VideoProcessors \
    Hog \
    VideoWidgets \
    VideoGui

HogApp.subdir = $$SRC_DIR/HogApp
VideoProcessors.subdir = $$SRC_DIR/VideoProcessors
Hog.subdir = $$SRC_DIR/Hog
VideoWidgets.subdir = $$SRC_DIR/VideoWidgets
VideoGui.subdir = $$SRC_DIR/VideoGui

VideoProcessors.depends = \
    Hog

HogApp.depends = \
    VideoProcessors \
    VideoWidgets \
    VideoGui

