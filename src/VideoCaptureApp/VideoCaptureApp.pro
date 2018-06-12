QT += core gui widgets
TARGET = VideoCaptureApp
TEMPLATE = app

include($$PWD/../tracking.pri)

DEPENDENCIES = VideoProcessors VideoWidgets VideoGui
INCLUDEPATH += $$addIncludes($$DEPENDENCIES)
LIBS += $$addLibs($$DEPENDENCIES)
PRE_TARGETDEPS += $$addTargetDeps($$DEPENDENCIES)

SOURCES += \
    main.cpp \
    videocapturemainwin.cpp

HEADERS += \
    videocapturemainwin.h

FORMS += \
    videocapturemainwin.ui
