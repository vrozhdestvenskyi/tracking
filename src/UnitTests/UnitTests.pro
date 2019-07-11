QT -= gui
QT += core widgets
TARGET = UnitTests
TEMPLATE = app

include($$PWD/../tracking.pri)
include($$PWD/../opencv.pri)
include($$PWD/../googletest.pri)

INCLUDEPATH += \
    $$OCV_ROOT_DIR/include \
    $$GTEST_INCLUDE_DIR
LIBS += \
    $$addLibsOcv(core highgui imgproc features2d calib3d)
    $$addLibsGtest(gtest)

DEPENDENCIES = Hog
INCLUDEPATH += $$addIncludes($$DEPENDENCIES)
LIBS += $$addLibs($$DEPENDENCIES)
PRE_TARGETDEPS += $$addTargetDeps($$DEPENDENCIES)

SOURCES += \
    main.cpp \
    labtest.cpp
