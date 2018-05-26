TARGET = VideoCaptureAppSln
TEMPLATE = subdirs

include($$PWD/../../src/tracking.pri)

SUBDIRS = \
    VideoWidgets \
    VideoGui \
    VideoProcessors \
    VideoCaptureApp

VideoWidgets.subdir = $$projectSrcDir(VideoWidgets)
VideoGui.subdir = $$projectSrcDir(VideoGui)
VideoProcessors.subdir = $$projectSrcDir(VideoProcessors)
VideoCaptureApp.subdir = $$projectSrcDir(VideoCaptureApp)

VideoCaptureApp.depends = \
    VideoWidgets \
    VideoGui \
    VideoProcessors
