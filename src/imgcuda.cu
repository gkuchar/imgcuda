// imgcuda.cu -- CUDA implementations of image processing functions.
// This file contains the CUDA implementations of the image processing functions declared in imgcuda.h.
// The functions in this file are optimized for performance on NVIDIA GPUs, and may use CUDA-specific features
// such as shared memory, texture memory, and CUDA kernels.

#include "ppm.h"
#include "imgcuda.h"
#include <cuda_runtime.h>

#define CHANNELS 3

__global__ void drop_red_kernel(unsigned char* data, int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height) {
        int offset = y * width + x;
        data[CHANNELS * offset] = 0;
    }
}

__global__ void drop_green_kernel(unsigned char* data, int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height) {
        int offset = y * width + x;
        data[CHANNELS * offset + 1] = 0;
    }
}

__global__ void drop_blue_kernel(unsigned char* data, int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height) {
        int offset = y * width + x;
        data[CHANNELS * offset + 2] = 0;
    }
}

__global__ void grayscale_kernel(unsigned char* data, int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height) {
        int offset = y * width + x;

        unsigned char r = data[CHANNELS * offset]; // red value for pixel
        unsigned char g = data[CHANNELS * offset + 1]; // green value for pixel
        unsigned char b = data[CHANNELS * offset + 2]; // blue value for pixel

        float gray_val = 0.21f*r + 0.71f*g + 0.07f*b;
        data[CHANNELS * offset] = (unsigned char)gray_val;
        data[CHANNELS * offset + 1] = (unsigned char)gray_val;
        data[CHANNELS * offset + 2] = (unsigned char)gray_val;
    }
}

__global__ void blur_kernel(unsigned char* data, int width, int height, int radius) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height) {
        for (int c = 0; c < 3; c++) {
            int pix_vals = 0;
            int pixels = 0;
            for (int blurRow = -radius; blurRow < radius + 1; ++blurRow) {
                for (int blurCol = -radius; blurCol < radius + 1; ++blurCol) {
                    int curRow = y + blurRow;
                    int curCol = x + blurCol;
                    if (curRow > -1 && curRow < height && curCol > -1 && curCol < width) {
                        pix_vals += data[CHANNELS * (curRow * width + curCol) + c];
                        pixels++;
                    }
                }
            }
            data[CHANNELS * (y * width + x) + c] = (unsigned char)min(255, max(0, pix_vals / pixels));
        }
    }
}

namespace imgcuda {
    ppm::Image drop_red(const ppm::Image& in, imgcuda::Timing& t) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float ms;

        // 1. Allocate device memory
        unsigned char* d_data;
        cudaError_t err = cudaMalloc(&d_data, in.bytes());
        if (err != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(err));

        // 2. Copy input to device
        cudaEventRecord(start);
        err = cudaMemcpy(d_data, in.data.data(), in.bytes(), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(err));
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        t.htd += ms;

        // 3. Launch kernel
        dim3 block(16, 16);
        dim3 grid((in.width + block.x - 1) / block.x, (in.height + block.y - 1) / block.y);
        cudaEventRecord(start);
        drop_red_kernel<<<grid, block>>>(d_data, in.width, in.height);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        t.kernel += ms;

        // 4. Copy result back
        ppm::Image out = in; // copy metadata (width, height)
        cudaEventRecord(start);
        err = cudaMemcpy(out.data.data(), d_data, out.bytes(), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(err));
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        t.dth += ms;

        // 5. Free device memory
        err = cudaFree(d_data);
        if (err != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(err));

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        return out;
    }

    // original API wraps timing version and discards metrics
    ppm::Image drop_red(const ppm::Image& in) {
        imgcuda::Timing t;
        return drop_red(in, t);
    }

    ppm::Image drop_green(const ppm::Image& in, imgcuda::Timing& t) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float ms;

        // 1. Allocate device memory
        unsigned char* d_data;
        cudaError_t err = cudaMalloc(&d_data, in.bytes());
        if (err != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(err));

        // 2. Copy input to device
        cudaEventRecord(start);
        err = cudaMemcpy(d_data, in.data.data(), in.bytes(), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(err));
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        t.htd += ms;

        // 3. Launch kernel
        dim3 block(16, 16);
        dim3 grid((in.width + block.x - 1) / block.x, (in.height + block.y - 1) / block.y);
        cudaEventRecord(start);
        drop_green_kernel<<<grid, block>>>(d_data, in.width, in.height);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        t.kernel += ms;

        // 4. Copy result back
        ppm::Image out = in; // copy metadata (width, height)
        cudaEventRecord(start);
        err = cudaMemcpy(out.data.data(), d_data, out.bytes(), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(err));
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        t.dth += ms;

        // 5. Free device memory
        err = cudaFree(d_data);
        if (err != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(err));

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        return out;
    }

    // original API wraps timing version and discards metrics
    ppm::Image drop_green(const ppm::Image& in) {
        imgcuda::Timing t;
        return drop_green(in, t);
    }

    ppm::Image drop_blue(const ppm::Image& in, imgcuda::Timing& t) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float ms;

        // 1. Allocate device memory
        unsigned char* d_data;
        cudaError_t err = cudaMalloc(&d_data, in.bytes());
        if (err != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(err));

        // 2. Copy input to device
        cudaEventRecord(start);
        err = cudaMemcpy(d_data, in.data.data(), in.bytes(), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(err));
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        t.htd += ms;

        // 3. Launch kernel
        dim3 block(16, 16);
        dim3 grid((in.width + block.x - 1) / block.x, (in.height + block.y - 1) / block.y);
        cudaEventRecord(start);
        drop_blue_kernel<<<grid, block>>>(d_data, in.width, in.height);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        t.kernel += ms;

        // 4. Copy result back
        ppm::Image out = in; // copy metadata (width, height)
        cudaEventRecord(start);
        err = cudaMemcpy(out.data.data(), d_data, out.bytes(), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(err));
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        t.dth += ms;

        // 5. Free device memory
        err = cudaFree(d_data);
        if (err != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(err));

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        return out;
    }

    // original API wraps timing version and discards metrics
    ppm::Image drop_blue(const ppm::Image& in) {
        imgcuda::Timing t;
        return drop_blue(in, t);
    }

    ppm::Image grayscale(const ppm::Image& in, imgcuda::Timing& t) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float ms;

        // 1. Allocate device memory
        unsigned char* d_data;
        cudaError_t err = cudaMalloc(&d_data, in.bytes());
        if (err != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(err));

        // 2. Copy input to device
        cudaEventRecord(start);
        err = cudaMemcpy(d_data, in.data.data(), in.bytes(), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(err));
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        t.htd += ms;

        // 3. Launch kernel
        dim3 block(16, 16);
        dim3 grid((in.width + block.x - 1) / block.x, (in.height + block.y - 1) / block.y);
        cudaEventRecord(start);
        grayscale_kernel<<<grid, block>>>(d_data, in.width, in.height);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        t.kernel += ms;

        // 4. Copy result back
        ppm::Image out = in; // copy metadata (width, height)
        cudaEventRecord(start);
        err = cudaMemcpy(out.data.data(), d_data, out.bytes(), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(err));
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        t.dth += ms;

        // 5. Free device memory
        err = cudaFree(d_data);
        if (err != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(err));

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        return out;
    }

    // original API wraps timing version and discards metrics
    ppm::Image grayscale(const ppm::Image& in) {
        imgcuda::Timing t;
        return grayscale(in, t);
    }

    ppm::Image blur(const ppm::Image& in, int radius, imgcuda::Timing& t) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float ms;

        // 1. Allocate device memory
        unsigned char* d_data;
        cudaError_t err = cudaMalloc(&d_data, in.bytes());
        if (err != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(err));

        // 2. Copy input to device
        cudaEventRecord(start);
        err = cudaMemcpy(d_data, in.data.data(), in.bytes(), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(err));
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        t.htd += ms;

        // 3. Launch kernel
        dim3 block(16, 16);
        dim3 grid((in.width + block.x - 1) / block.x, (in.height + block.y - 1) / block.y);
        cudaEventRecord(start);
        blur_kernel<<<grid, block>>>(d_data, in.width, in.height, radius);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        t.kernel += ms;

        // 4. Copy result back
        ppm::Image out = in; // copy metadata (width, height)
        cudaEventRecord(start);
        err = cudaMemcpy(out.data.data(), d_data, out.bytes(), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(err));
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        t.dth += ms;

        // 5. Free device memory
        err = cudaFree(d_data);
        if (err != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(err));

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        return out;
    }

    // original API wraps timing version and discards metrics
    ppm::Image blur(const ppm::Image& in, int radius) {
        imgcuda::Timing t;
        return blur(in, radius, t);
    }

    ppm::Image sharpen(const ppm::Image& in, Timing& t)   { return in; } // TODO
    ppm::Image sobel_x(const ppm::Image& in, Timing& t)   { return in; } // TODO
    ppm::Image sobel_y(const ppm::Image& in, Timing& t)   { return in; } // TODO
}