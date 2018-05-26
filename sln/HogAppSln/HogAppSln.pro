TARGET = HogAppSln
TEMPLATE = subdirs

include($$PWD/../../src/tracking.pri)

SUBDIRS = \
    VideoWidgets \
    VideoGui \
    VideoProcessors \
    Hog \
    HogApp

VideoWidgets.subdir = $$projectSrcDir(VideoWidgets)
VideoGui.subdir = $$projectSrcDir(VideoGui)
VideoProcessors.subdir = $$projectSrcDir(VideoProcessors)
Hog.subdir = $$projectSrcDir(Hog)
HogApp.subdir = $$projectSrcDir(HogApp)

HogApp.depends = \
    VideoWidgets \
    VideoGui \
    Hog \
    VideoProcessors
