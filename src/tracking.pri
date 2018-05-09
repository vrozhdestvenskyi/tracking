BIN_DIR = $$PWD/../bin/
SRC_DIR = $$PWD/../src/

defineReplace(projectBinDir) {
    projectName = $$1
    CONFIG(debug, debug|release) {
        return($$BIN_DIR/$$projectName/debug/)
    } else {
        return($$BIN_DIR/$$projectName/release/)
    }
}

DESTDIR = $$projectBinDir($$TARGET)
OBJECTS_DIR = $$DESTDIR
MOC_DIR = $$DESTDIR
RCC_DIR = $$DESTDIR
UI_DIR = $$DESTDIR
