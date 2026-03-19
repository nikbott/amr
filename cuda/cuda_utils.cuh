#pragma once

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error)); \
        } \
    } while(0)

// Kernel launch error checking
#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

// Device synchronization with error checking
#define CUDA_SYNC() CUDA_CHECK(cudaDeviceSynchronize())

namespace cuda_utils {

// Get optimal block size for a kernel
inline int getOptimalBlockSize() {
    return 256; // Good default for most modern GPUs
}

// Calculate grid size given total threads and block size
inline int getGridSize(int n, int blockSize) {
    return (n + blockSize - 1) / blockSize;
}

// Query GPU properties
inline void printGPUInfo() {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        
        printf("GPU %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total Global Memory: %.2f GB\n", 
               prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    }
}

} // namespace cuda_utils

#endif // USE_CUDA
