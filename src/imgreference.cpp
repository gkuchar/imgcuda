// imgreference.cpp -- CPU reference implementations of image processing functions.
// These functions are not optimized for performance, but are intended to be correct and easy to understand. 
// They can be used for testing and validation of the GPU implementations in imgcuda.cu.

#include "ppm.h"
#include "imgreference.h"

namespace imgcpu {
    ppm::Image drop_red(const ppm::Image& in) {
        int i;
        ppm::Image out = in;
        for (i = 0; i < in.bytes(); i = i + 3) {
            out.data[i] = 0;
        }
        return out;
    }

    ppm::Image drop_green(const ppm::Image& in) {
        int i;
        ppm::Image out = in;
        for (i = 1; i < in.bytes(); i = i + 3) {
            out.data[i] = 0;
        }
        return out;
    }

    ppm::Image drop_blue(const ppm::Image& in) {
        int i;
        ppm::Image out = in;
        for (i = 2; i < in.bytes(); i = i + 3) {
            out.data[i] = 0;
        }
        return out;
    }
}