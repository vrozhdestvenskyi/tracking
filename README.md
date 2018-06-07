This code was written in QtCreator 4.6.1 with MinGW 5.3.0 compiler under Windows 10. It depends on OpenCV-2.4.13 library. Path to its binaries is specified in ./src/opencv.pri file.

To compile it in QtCreator, please import subdirs-projects from ./sln subdirectory. Please note that directory with binaries is specified in ./src/tracking.pri file. There is no need to use Shadow build.

For algorithms testing, the data from VOT2016 dataset is used. It can be downloaded here: http://www.votchallenge.net/vot2016/dataset.html