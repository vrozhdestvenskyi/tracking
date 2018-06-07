CONFIG -= debug_and_release debug_and_release_target

SRC_DIR = $$PWD

BIN_DIR = $$SRC_DIR/../bin
CONFIG(debug, debug|release) {
    BIN_DIR = $$BIN_DIR-debug
} else {
    BIN_DIR = $$BIN_DIR-release
}

defineReplace(addIncludes) {
    libs = $$1
    res = ""
    for(lib, libs) {
        res = $${res} $${SRC_DIR}/$${lib}
    }
    return($$res)
}

defineReplace(addLibs) {
    libs = $$1
    res = ""
    for(lib, libs) {
        res = $${res} -L$${BIN_DIR}/$${lib} -l$${lib}
    }
    return($$res)
}

defineReplace(addTargetDeps) {
    deps = $$1
    res = ""
    for(dep, deps) {
        res = $${res} $${BIN_DIR}/$${dep}/lib$${dep}.a
    }
    return($$res)
}

DESTDIR = $$BIN_DIR/$$TARGET
OBJECTS_DIR = $$DESTDIR
MOC_DIR = $$DESTDIR
RCC_DIR = $$DESTDIR
UI_DIR = $$DESTDIR
