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

namespace imgcuda {
    ppm::Image drop_red(const ppm::Image& in) {
        // 1. Allocate device memory
        unsigned char* d_data;
        cudaError_t err = cudaMalloc(&d_data, in.bytes());
        if (err != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(err));

        // 2. Copy input to device
        err = cudaMemcpy(d_data, in.data.data(), in.bytes(), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(err));

        // 3. Launch kernel
        dim3 block(16, 16);
        dim3 grid((in.width + block.x - 1) / block.x, (in.height + block.y - 1) / block.y);
        drop_red_kernel<<<grid, block>>>(d_data, in.width, in.height);

        // 4. Copy result back
        ppm::Image out = in; // copy metadata (width, height)
        err = cudaMemcpy(out.data.data(), d_data, out.bytes(), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(err));

        // 5. Free device memory
        err = cudaFree(d_data);
        if (err != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(err));
        return out;
    }

    ppm::Image drop_green(const ppm::Image& in) {
        // 1. Allocate device memory
        unsigned char* d_data;
        cudaError_t err = cudaMalloc(&d_data, in.bytes());
        if (err != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(err));

        // 2. Copy input to device
        err = cudaMemcpy(d_data, in.data.data(), in.bytes(), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(err));

        // 3. Launch kernel
        dim3 block(16, 16);
        dim3 grid((in.width + block.x - 1) / block.x, (in.height + block.y - 1) / block.y);
        drop_green_kernel<<<grid, block>>>(d_data, in.width, in.height);

        // 4. Copy result back
        ppm::Image out = in; // copy metadata (width, height)
        err = cudaMemcpy(out.data.data(), d_data, out.bytes(), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(err));

        // 5. Free device memory
        err = cudaFree(d_data);
        if (err != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(err));
        return out;
    }

    ppm::Image drop_blue(const ppm::Image& in) {
        // 1. Allocate device memory
        unsigned char* d_data;
        cudaError_t err = cudaMalloc(&d_data, in.bytes());
        if (err != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(err));

        // 2. Copy input to device
        err = cudaMemcpy(d_data, in.data.data(), in.bytes(), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(err));

        // 3. Launch kernel
        dim3 block(16, 16);
        dim3 grid((in.width + block.x - 1) / block.x, (in.height + block.y - 1) / block.y);
        drop_blue_kernel<<<grid, block>>>(d_data, in.width, in.height);

        // 4. Copy result back
        ppm::Image out = in; // copy metadata (width, height)
        err = cudaMemcpy(out.data.data(), d_data, out.bytes(), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(err));

        // 5. Free device memory
        err = cudaFree(d_data);
        if (err != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(err));
        return out;
    }
}