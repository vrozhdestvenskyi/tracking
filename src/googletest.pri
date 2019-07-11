GTEST_INCLUDE_DIR = C:/googletest/source/googletest/include
GTEST_BIN_DIR = C:/googletest/build/bin

defineReplace(addLibsGtest) {
    libs = $$1
    res = ""
    for(lib, libs) {
        res = $${res} -L$${GTEST_BIN_DIR} -l$${lib}
    }
    return($$res)
}

