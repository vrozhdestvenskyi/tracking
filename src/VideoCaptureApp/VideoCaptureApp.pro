QT += core gui widgets
TARGET = VideoCaptureApp
TEMPLATE = app

include($$PWD/../tracking.pri)

# OpenCL
INCLUDEPATH += "C:/Intel/OpenCL/sdk/include"
LIBS += -L"C:/Intel/OpenCL/sdk/bin/icd/x86" -lOpenCL

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
