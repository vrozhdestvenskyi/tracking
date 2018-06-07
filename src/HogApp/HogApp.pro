QT += core gui widgets
TARGET = HogApp
TEMPLATE = app

include($$PWD/../tracking.pri)
include($$PWD/../opencv.pri)

INCLUDEPATH += $$OCV_ROOT_DIR/include
LIBS += $$addLibsOcv(core highgui imgproc features2d calib3d)

DEPENDENCIES = VideoProcessors Hog VideoWidgets VideoGui
INCLUDEPATH += $$addIncludes($$DEPENDENCIES)
LIBS += $$addLibs($$DEPENDENCIES)
PRE_TARGETDEPS += $$addTargetDeps($$DEPENDENCIES)

SOURCES += \
    main.cpp \
    hogmainwin.cpp

HEADERS += \
    hogmainwin.h

FORMS += \
    hogmainwin.ui
