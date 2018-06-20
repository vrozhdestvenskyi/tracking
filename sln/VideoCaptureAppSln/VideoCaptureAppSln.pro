TARGET = VideoCaptureAppSln
TEMPLATE = subdirs

include($$PWD/../../src/tracking.pri)

SUBDIRS += \
    VideoCaptureApp \
    VideoProcessors \
    VideoWidgets \
    VideoGui

VideoCaptureApp.subdir = $$SRC_DIR/VideoCaptureApp
VideoProcessors.subdir = $$SRC_DIR/VideoProcessors
VideoWidgets.subdir = $$SRC_DIR/VideoWidgets
VideoGui.subdir = $$SRC_DIR/VideoGui

VideoGui.depends = \
    VideoProcessors

VideoCaptureApp.depends = \
    VideoWidgets \
    VideoGui
