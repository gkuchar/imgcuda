// imgreference.cpp -- CPU reference implementations of image processing functions.
// These functions are not optimized for performance, but are intended to be correct and easy to understand. 
// They can be used for testing and validation of the GPU implementations in imgcuda.cu.

#include "ppm.h"
#include "imgreference.h"
#include <algorithm>
using std::min;
using std::max;

#define CHANNELS 3

namespace imgcpu {
    ppm::Image drop_red(const ppm::Image& in) {
        int i;
        ppm::Image out = in;
        for (i = 0; i < (int)in.bytes(); i = i + 3) {
            out.data[i] = 0;
        }
        return out;
    }

    ppm::Image drop_green(const ppm::Image& in) {
        int i;
        ppm::Image out = in;
        for (i = 1; i < (int)in.bytes(); i = i + 3) {
            out.data[i] = 0;
        }
        return out;
    }

    ppm::Image drop_blue(const ppm::Image& in) {
        int i;
        ppm::Image out = in;
        for (i = 2; i < (int)in.bytes(); i = i + 3) {
            out.data[i] = 0;
        }
        return out;
    }

    ppm::Image grayscale(const ppm::Image& in) {
        int i = 0;
        ppm::Image out = in;
        while (i < (int)in.bytes()) {
            unsigned char r = in.data[i];
            unsigned char g = in.data[i + 1];
            unsigned char b = in.data[i + 2];

            float gray_val = 0.21f*r + 0.71f*g + 0.07f*b;

            out.data[i] = (unsigned char)gray_val;
            out.data[i + 1] = (unsigned char)gray_val;
            out.data[i + 2] = (unsigned char)gray_val;

            i = i + 3;
        }
        return out;
    }

    ppm::Image blur(const ppm::Image& in, int radius) {
        ppm::Image out = in;
        for (int y = 0; y < in.height; y++) {
            for (int x = 0; x < in.width; x++) {
                for (int c = 0; c < 3; c++) {
                    int pix_vals = 0;
                    int pixels = 0;
                    for (int blurRow = -radius; blurRow < radius + 1; blurRow++) {
                        for (int blurCol = -radius; blurCol < radius + 1; blurCol++) {
                            int curRow = y + blurRow;
                            int curCol = x + blurCol;
                            if (curRow > -1 && curRow < in.height && curCol > -1 && curCol < in.width) {
                                pix_vals += in.data[CHANNELS * (curRow * in.width + curCol) + c];
                                pixels++;
                            }
                        }
                    }
                    out.data[CHANNELS * (y * in.width + x) + c] = (unsigned char)min(255, max(0, pix_vals / pixels));
                }
            }
        }
        return out;
    }

    ppm::Image sharpen(const ppm::Image& in)   { return in; } // TODO
    ppm::Image sobel_x(const ppm::Image& in)   { return in; } // TODO
    ppm::Image sobel_y(const ppm::Image& in)   { return in; } // TODO
}