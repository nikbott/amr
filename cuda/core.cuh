/**
 * @file core.cuh
 * @brief CUDA Core utilities: Morton Codes and Device Intrinsics.
 */
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "../common/core.hpp"  // amr::MortonCode, Coordinate, morton::, constants
                               // (morton:: ops are __host__ __device__ via AMR_HD)

// Backend-local alias kept for the other cuda/ sources (tree.cuh, physics.cuh,
// tests.cu) that spell the qualifier as HOST_DEVICE.
#define HOST_DEVICE __host__ __device__