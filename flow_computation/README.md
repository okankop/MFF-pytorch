# Flow Computation
Optical flow extraction code using OpenCV. The code is taken from the offical OpenCV documentation. Currently Brox algorithm is used for GPU. Optical flow frames are scaled according to the maximum value appeared in absolute values of horizontal and vertical components and mapped discretely into the interval [0, 250]. By means of this step, the range of optical flow frames becomes same as RGB images.

## Compilation
Run `./compile` in order to compile.

## Help
Type `./optical_flow -h` to see the help message below.  
