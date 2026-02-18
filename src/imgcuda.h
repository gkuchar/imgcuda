// imgcuda.h -- CUDA implementations of image processing functions.

// These functions are declared here and defined in imgcuda.cu. 

// They are intended to be called from CPU code (e.g. main.cpp) and 
// will launch GPU kernels to perform the actual work.

// The function signatures and behavior should be designed to be as 
// simple and intuitive as possible for CPU callers, while still 
// allowing for efficient GPU implementations. 

// The GPU kernels themselves should be implemented in a way that is 
// efficient and takes advantage of CUDA features, but the CPU-facing API should abstract away