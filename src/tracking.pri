CONFIG -= debug_and_release debug_and_release_target

SRC_DIR = $$PWD

BIN_DIR = $$SRC_DIR/../bin
CONFIG(debug, debug|release) {
    BIN_DIR = $$BIN_DIR-debug
} else {
    BIN_DIR = $$BIN_DIR-release
}

defineReplace(projectSrcDir) {
    projectName = $$1
    return($$SRC_DIR/$$projectName)
}

defineReplace(projectBinDir) {
    projectName = $$1
    return($$BIN_DIR/$$projectName)
}

DESTDIR = $$projectBinDir($$TARGET)
OBJECTS_DIR = $$DESTDIR
MOC_DIR = $$DESTDIR
RCC_DIR = $$DESTDIR
UI_DIR = $$DESTDIR
