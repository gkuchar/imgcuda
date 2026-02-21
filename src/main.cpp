// main.cpp -- Main program for image processing using CUDA.
// This program serves as the entry point for the image processing application.
// It will read an input image, perform some processing using CUDA, and write the output image
// The specific processing steps will be defined in the imgcuda.h and imgcuda.cu files, and may include
// operations such as color inversion, blurring, edge detection, etc.
// The main function will handle command-line arguments, call the appropriate functions from imgcuda.h,
// and manage any necessary resources (e.g. memory allocation, error handling, etc.).

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "ppm.h"
#include "imgcuda.h"
#include "imgreference.h"   
#include <functional>
#include <map>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

int main(int argc, char* argv[]) {
    int runs = 10;
    std::string input, output, filter, filter_full;
    int radius = 1;
    bool verify = false;

    std::map<std::string, std::function<ppm::Image(const ppm::Image&)>> gpu_filters = {
    {"drop_red",   imgcuda::drop_red},
    {"drop_green", imgcuda::drop_green},
    {"drop_blue",  imgcuda::drop_blue},
    {"grayscale",  imgcuda::grayscale},
    {"sharpen",    imgcuda::sharpen},
    {"sobel_x",    imgcuda::sobel_x},
    {"sobel_y",    imgcuda::sobel_y},
    };

    std::map<std::string, std::function<ppm::Image(const ppm::Image&)>> cpu_filters = {
    {"drop_red",   imgcpu::drop_red},
    {"drop_green", imgcpu::drop_green},
    {"drop_blue",  imgcpu::drop_blue},
    {"grayscale",  imgcpu::grayscale},
    {"sharpen",    imgcpu::sharpen},
    {"sobel_x",    imgcpu::sobel_x},
    {"sobel_y",    imgcpu::sobel_y},
    };

    cudaEvent_t start, stop;
    float htdTotal = 0;
    float kernExecTotal = 0;
    float dthTotal = 0;

    if (argc < 3) {
        printf("Usage: %s input.ppm output.ppm --filter <name> [--radius <r>] [--runs <n>] [--verify]\n", argv[0]);
        exit(1);
    }

    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--filter")       filter  = argv[++i];
        else if (arg == "--radius")  radius  = std::stoi(argv[++i]);
        else if (arg == "--runs")    runs    = std::stoi(argv[++i]);
        else if (arg == "--verify")  verify  = true;
    }

    if (filter.empty()) {
        printf("Error: --filter is required\n");
        exit(1);
    }
    if (gpu_filters.count(filter) == 0 && filter != "blur") {
        printf("Error: unknown filter '%s'\n", filter.c_str());
        exit(1);
    }
    if (filter == "blur" && (radius < 1 || radius > 5)) {
        printf("Error: --radius must be between 1 and 5\n");
        exit(1);
    }
    if (runs < 10 || runs > 25) {
        printf("Error: --runs must be between 10 and 25\n");
        exit(1);
    }

    input = argv[1];
    output = argv[2];

    ppm::Image in_image = ppm::read(input);
    ppm::Image out_image;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    if (filter == "blur") {
        out_image = imgcuda::blur(in_image, radius);
        filter_full = std::string("blur") + " radius=" + std::to_string(radius);
    }
    else {
        out_image = gpu_filters[filter](in_image);
        filter_full = filter;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    ppm::write(output, out_image);

    printf("\nFilter: %s\n", filter_full.c_str());
    printf("Runs: %d\n", runs);
    printf("Average H2D: %f ms\n", htdTotal / runs);
    printf("Average Kernel: %f ms\n", kernExecTotal / runs);
    printf("Average D2H: %f ms\n", dthTotal / runs);

    return 0;
}