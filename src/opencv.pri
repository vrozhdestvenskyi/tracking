OCV_ROOT_DIR = C:/opencv-2.4.13/build/install
OCV_BIN_DIR = $$OCV_ROOT_DIR/x86/mingw/bin/

defineReplace(ocvLibName) {
    libName = $$1
    return("opencv_"$$libName"2413")
}
