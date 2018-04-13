#!/bin/sh

CODEFILE=${1:-'optical_flow.cpp'}
BINARYFILE=${2:-'optical_flow'}

OPENCV_INSTALL="/usr/local/"
# OPENCV_INSTALL="/usr/local/"


g++ -std=c++11 -I${OPENCV_INSTALL}include -L${OPENCV_INSTALL}lib -g -o $BINARYFILE $CODEFILE -lopencv_calib3d -lopencv_core -lopencv_cudaarithm -lopencv_cudabgsegm -lopencv_cudacodec -lopencv_cudafeatures2d -lopencv_cudafilters -lopencv_cudaimgproc -lopencv_cudalegacy -lopencv_objdetect -lopencv_cudaoptflow -lopencv_cudastereo -lopencv_cudawarping -lopencv_cudev -lopencv_features2d -lopencv_flann -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_shape -lopencv_stitching -lopencv_superres -lopencv_videoio -lopencv_video -lopencv_videostab
