// Similar functions to imgcuda.h, but implemented in CPU code for reference and testing purposes.
//Different namespace.

#pragma once
#include "ppm.h"

namespace imgcpu {
    ppm::Image drop_red(const ppm::Image& in);
    ppm::Image drop_green(const ppm::Image& in);
    ppm::Image drop_blue(const ppm::Image& in);
    ppm::Image grayscale(const ppm::Image& in);
    
    ppm::Image blur(const ppm::Image& in, int radius);
    ppm::Image sharpen(const ppm::Image& in);

    ppm::Image sobel_x(const ppm::Image& in);
    ppm::Image sobel_y(const ppm::Image& in);
}