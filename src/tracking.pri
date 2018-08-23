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

defineReplace(copyToDestDir) {
    files = $$1
#    message(AAA)
#    message($$files)
    LINK = ""
    for(FILE, files) {
        DDIR = $$DESTDIR
        message($$FILE)
        message($$DDIR)
        message(AAA)
        win32:FILE ~= s,/,\\,g
        win32:DDIR ~= s,/,\\,g
        message($$FILE)
        message($$DDIR)
        LINK += $$QMAKE_COPY $$quote($$FILE) $$quote($$DDIR) $$escape_expand(\\n\\t)
    }
    message($$LINK)
    #export(QMAKE_POST_LINK)
    return($$LINK)
}

defineReplace(copyToDestDir2) {
    copydata.commands = $$copyToDestDir($$1)
    first.depends = $$(first) copydata
    export(first.depends)
    export(copydata.commands)
    QMAKE_EXTRA_TARGETS += first copydata
    export(QMAKE_EXTRA_TARGETS)
}

DESTDIR = $$BIN_DIR/$$TARGET
OBJECTS_DIR = $$DESTDIR
MOC_DIR = $$DESTDIR
RCC_DIR = $$DESTDIR
UI_DIR = $$DESTDIR
