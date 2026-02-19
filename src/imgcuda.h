// imgcuda.h -- CUDA implementations of image processing functions.

// These functions are declared here and defined in imgcuda.cu. 

// They are intended to be called from CPU code (e.g. main.cpp) and 
// will launch GPU kernels to perform the actual work.

// The function signatures and behavior should be designed to be as 
// simple and intuitive as possible for CPU callers, while still 
// allowing for efficient GPU implementations. 

// The GPU kernels themselves should be implemented in a way that is 
// efficient and takes advantage of CUDA features, but the CPU-facing API should abstract away

#pragma once
#include "ppm.h"

namespace imgcuda {
    ppm::Image drop_red(const ppm::Image& in);
    ppm::Image drop_green(const ppm::Image& in);
    ppm::Image drop_blue(const ppm::Image& in);
    ppm::Image grayscale(const ppm::Image& in);
    
    ppm::Image blur(const ppm::Image& in, int radius);
    ppm::Image sharpen(const ppm::Image& in);

    ppm::Image sobel_x(const ppm::Image& in);
    ppm::Image sobel_y(const ppm::Image& in);
 }