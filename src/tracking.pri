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

defineReplace(generateCopyingCommand) {
    file_paths = $$1
    dest_dir = $$DESTDIR
    cmd = ""
    for(path, file_paths) {
        win32:path ~= s,/,\\,g
        win32:dest_dir ~= s,/,\\,g
        cmd += $$QMAKE_COPY $$quote($$path) $$quote($$dest_dir) $$escape_expand(\\n\\t)
    }
    return($$cmd)
}

defineTest(copyFilesToDestDir) {
    file_paths = $$1
    copydata.commands = $$generateCopyingCommand($$file_paths)
    first.depends = $(first) copydata
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
