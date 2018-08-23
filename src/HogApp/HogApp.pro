QT += core gui widgets
TARGET = HogApp
TEMPLATE = app

include($$PWD/../tracking.pri)
include($$PWD/../opencl.pri)
include($$PWD/../opencv.pri)

INCLUDEPATH += $$OCL_INCLUDE_DIR
LIBS += $$OCL_LIB

# For Piotr's HOG implementation
CONFIG += mmx sse sse2
QMAKE_FLAGS += -msse4.1 -mssse3 -msse3 -msse2 -msse
QMAKE_CXXFLAGS += -msse4.1 -mssse3 -msse3 -msse2 -msse

INCLUDEPATH += $$OCV_ROOT_DIR/include
LIBS += $$addLibsOcv(core highgui imgproc features2d calib3d)

DEPENDENCIES = VideoProcessors Hog HogPiotr VideoWidgets VideoGui
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

#copydata.commands = $$copyToDestDir($${SRC_DIR}/Hog/hog.cl)
#first.depends = $(first) copydata
#export(first.depends)
#export(copydata.commands)
#QMAKE_EXTRA_TARGETS += first copydata

$$copyToDestDir2($${SRC_DIR}/Hog/hog.cl)
