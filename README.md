This code was written in QtCreator 4.6.1 with MinGW 5.3.0 compiler under Windows 10. Some of the algorithms are compared against existing OpenCV implementations, that's why corresponding test applications depend from OpenCV-2.4.13 library. Path to its binaries is specified in ./src/opencv.pri file.
Also this code uses OpenCL 1.2.

To compile any application in QtCreator, please import a corresponding subdirs-project from ./sln subdirectory. Please note that directory with binaries is specified in ./src/tracking.pri file, so there is no need to use Shadow build.

For algorithms testing, the VOT2016 database is used. It can be downloaded here: http://www.votchallenge.net/vot2016/dataset.html