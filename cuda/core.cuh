/**
 * @file core.cuh
 * @brief CUDA Core utilities: Morton Codes and Device Intrinsics.
 */
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#define HOST_DEVICE __host__ __device__

namespace amr {

// --- Geometric Constants ---
namespace constants {
    constexpr double SQRT3_OVER_2 = 0.86602540378;
    constexpr double SQRT2_OVER_2 = 0.70710678118;
}

// --- Strong Types ---
struct MortonCode {
    uint64_t value;
    HOST_DEVICE bool operator<(const MortonCode& other) const { return value < other.value; }
    HOST_DEVICE bool operator==(const MortonCode& other) const { return value == other.value; }
    HOST_DEVICE bool operator!=(const MortonCode& other) const { return value != other.value; }
};

struct Coordinate {
    uint32_t value;
    HOST_DEVICE double to_double() const { return static_cast<double>(value); }
};

// --- Morton Operations (SWAR for GPU) ---
namespace morton {
    // 2D Masks
    constexpr uint64_t MASK2_X = 0x5555555555555555;
    constexpr uint64_t MASK2_Y = 0xAAAAAAAAAAAAAAAA;

    // 3D Masks
    constexpr uint64_t MASK3_X = 0x9249249249249249;
    constexpr uint64_t MASK3_Y = 0x2492492492492492;
    constexpr uint64_t MASK3_Z = 0x4924924924924924;

    HOST_DEVICE inline uint64_t spread_bits_2d(uint32_t a) {
        uint64_t x = a;
        x = (x | (x << 32)) & 0x00000000FFFFFFFF;
        x = (x | (x << 16)) & 0x0000FFFF0000FFFF;
        x = (x | (x << 8))  & 0x00FF00FF00FF00FF;
        x = (x | (x << 4))  & 0x0F0F0F0F0F0F0F0F;
        x = (x | (x << 2))  & 0x3333333333333333;
        x = (x | (x << 1))  & 0x5555555555555555;
        return x;
    }

    HOST_DEVICE inline uint32_t compact_bits_2d(uint64_t x) {
        x &= 0x5555555555555555;
        x = (x ^ (x >> 1)) & 0x3333333333333333;
        x = (x ^ (x >> 2)) & 0x0F0F0F0F0F0F0F0F;
        x = (x ^ (x >> 4)) & 0x00FF00FF00FF00FF;
        x = (x ^ (x >> 8)) & 0x0000FFFF0000FFFF;
        x = (x ^ (x >> 16)) & 0x00000000FFFFFFFF;
        return (uint32_t)x;
    }

    HOST_DEVICE inline uint64_t spread_bits_3d(uint32_t a) {
        uint64_t x = a & 0x1FFFFF;
        x = (x | (x << 32)) & 0x1F00000000FFFF;
        x = (x | (x << 16)) & 0x1F0000FF0000FF;
        x = (x | (x << 8))  & 0x100F00F00F00F00F;
        x = (x | (x << 4))  & 0x10C30C30C30C30C3;
        x = (x | (x << 2))  & 0x1249249249249249;
        return x;
    }

    HOST_DEVICE inline uint32_t compact_bits_3d(uint64_t x) {
        x &= 0x1249249249249249;
        x = (x ^ (x >> 2))  & 0x10C30C30C30C30C3;
        x = (x ^ (x >> 4))  & 0x100F00F00F00F00F;
        x = (x ^ (x >> 8))  & 0x1F0000FF0000FF;
        x = (x ^ (x >> 16)) & 0x1F00000000FFFF;
        x = (x ^ (x >> 32)) & 0x1FFFFF;
        return (uint32_t)x;
    }

    HOST_DEVICE inline MortonCode encode_2d(Coordinate x, Coordinate y) {
        return { spread_bits_2d(x.value) | (spread_bits_2d(y.value) << 1) };
    }

    HOST_DEVICE inline void decode_2d(MortonCode code, Coordinate& x, Coordinate& y) {
        x.value = compact_bits_2d(code.value);
        y.value = compact_bits_2d(code.value >> 1);
    }

    HOST_DEVICE inline MortonCode encode_3d(Coordinate x, Coordinate y, Coordinate z) {
        return { spread_bits_3d(x.value) | (spread_bits_3d(y.value) << 1) | (spread_bits_3d(z.value) << 2) };
    }

    HOST_DEVICE inline void decode_3d(MortonCode code, Coordinate& x, Coordinate& y, Coordinate& z) {
        x.value = compact_bits_3d(code.value);
        y.value = compact_bits_3d(code.value >> 1);
        z.value = compact_bits_3d(code.value >> 2);
    }
}
}