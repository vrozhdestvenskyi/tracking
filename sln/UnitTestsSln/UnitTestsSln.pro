TARGET = UnitTestsSln
TEMPLATE = subdirs

include($$PWD/../../src/tracking.pri)

SUBDIRS += \
    UnitTests \
    ImgProc \
    HogPiotr \
    VideoProcessors

UnitTests.subdir = $$SRC_DIR/UnitTests
ImgProc.subdir = $$SRC_DIR/ImgProc
HogPiotr.subdir = $$SRC_DIR/HogPiotr
VideoProcessors.subdir = $$SRC_DIR/VideoProcessors

UnitTests.depends = \
    ImgProc \
    HogPiotr \
    VideoProcessors
