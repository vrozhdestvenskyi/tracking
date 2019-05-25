QT -= gui
QT += core widgets
TARGET = UnitTests
TEMPLATE = app

include($$PWD/../tracking.pri)
include($$PWD/../opencv.pri)

INCLUDEPATH += $$OCV_ROOT_DIR/include
LIBS += $$addLibsOcv(core highgui imgproc features2d calib3d)

DEPENDENCIES = Hog
INCLUDEPATH += $$addIncludes($$DEPENDENCIES)
LIBS += $$addLibs($$DEPENDENCIES)
PRE_TARGETDEPS += $$addTargetDeps($$DEPENDENCIES)

SOURCES += \
    main.cpp
