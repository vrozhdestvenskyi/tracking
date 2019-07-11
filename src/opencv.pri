OCV_ROOT_DIR = C:/opencv-2.4.13/build/install
OCV_BIN_DIR = $$OCV_ROOT_DIR/x86/mingw/bin/

defineReplace(addLibsOcv) {
    libs = $$1
    res = ""
    for(lib, libs) {
        res = $${res} -L$${OCV_BIN_DIR} -lopencv_$${lib}2413
    }
    return($$res)
}

