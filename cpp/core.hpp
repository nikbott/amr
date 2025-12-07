#pragma once
#include <cstdint>
#include <array>
#include <vector>
#include <algorithm>
#include <cmath>
#include <concepts>
#include <execution>
#include <numeric>

// Check for BMI2 instruction set support for fast bit-interleaving
#if defined(__BMI2__)
#include <immintrin.h>
#endif

namespace amr {

/**
 * @brief Morton Code (Z-order curve) operations for 2D and 3D.
 * Conforms to the bitwise interleaving strategies described in Burstedde et al.
 */
namespace morton {

    // 2D Masks for bit spreading
    static constexpr uint64_t MASK2_1 = 0x00000000FFFFFFFF;
    static constexpr uint64_t MASK2_2 = 0x0000FFFF0000FFFF;
    static constexpr uint64_t MASK2_3 = 0x00FF00FF00FF00FF;
    static constexpr uint64_t MASK2_4 = 0x0F0F0F0F0F0F0F0F;
    static constexpr uint64_t MASK2_5 = 0x3333333333333333;
    static constexpr uint64_t MASK2_6 = 0x5555555555555555;

    // 3D Masks for bit spreading
    static constexpr uint64_t MASK3_1 = 0x00000000001FFFFF;
    
    /**
     * @brief Spread bits for 2D Morton encoding.
     * Maps xxxxxxxx -> 0x0x0x0x0x0x0x0x
     */
    [[nodiscard]] constexpr uint64_t spread_bits_2d(uint64_t n) {
        n &= MASK2_1;
        n = (n | (n << 16)) & MASK2_2;
        n = (n | (n << 8))  & MASK2_3;
        n = (n | (n << 4))  & MASK2_4;
        n = (n | (n << 2))  & MASK2_5;
        n = (n | (n << 1))  & MASK2_6;
        return n;
    }

    /**
     * @brief Compact bits for 2D Morton decoding.
     * Maps 0x0x0x0x0x0x0x0x -> xxxxxxxx
     */
    [[nodiscard]] constexpr uint64_t compact_bits_2d(uint64_t n) {
        n &= MASK2_6;
        n = (n | (n >> 1))  & MASK2_5;
        n = (n | (n >> 2))  & MASK2_4;
        n = (n | (n >> 4))  & MASK2_3;
        n = (n | (n >> 8))  & MASK2_2;
        n = (n | (n >> 16)) & MASK2_1;
        return n;
    }

    /**
     * @brief Spread bits for 3D Morton encoding.
     * Maps xxxxx -> 00x00x00x00x00x
     */
    [[nodiscard]] constexpr uint64_t spread_bits_3d(uint64_t x) {
        x &= 0x1FFFFF; 
        x = (x | (x << 32)) & 0x1F00000000FFFF;
        x = (x | (x << 16)) & 0x1F0000FF0000FF;
        x = (x | (x << 8))  & 0x100F00F00F00F00F;
        x = (x | (x << 4))  & 0x10C30C30C30C30C3;
        x = (x | (x << 2))  & 0x1249249249249249;
        return x;
    }

    /**
     * @brief Compact bits for 3D Morton decoding.
     */
    [[nodiscard]] constexpr uint64_t compact_bits_3d(uint64_t x) {
        x &= 0x1249249249249249;
        x = (x ^ (x >> 2))  & 0x10C30C30C30C30C3;
        x = (x ^ (x >> 4))  & 0x100F00F00F00F00F;
        x = (x ^ (x >> 8))  & 0x1F0000FF0000FF;
        x = (x ^ (x >> 16)) & 0x1F00000000FFFF;
        x = (x ^ (x >> 32)) & 0x1FFFFF;
        return x;
    }

    /**
     * @brief Encodes 2D coordinates into a Morton code.
     */
    [[nodiscard]] inline uint64_t encode_2d(uint32_t x, uint32_t y) {
#if defined(__BMI2__)
        return _pdep_u64(x, MASK2_6) | _pdep_u64(y, 0xAAAAAAAAAAAAAAAA);
#else
        return (spread_bits_2d(y) << 1) | spread_bits_2d(x);
#endif
    }

    /**
     * @brief Decodes a Morton code into 2D coordinates.
     */
    [[nodiscard]] inline std::array<uint32_t, 2> decode_2d(uint64_t code) {
#if defined(__BMI2__)
        return { 
            static_cast<uint32_t>(_pext_u64(code, MASK2_6)), 
            static_cast<uint32_t>(_pext_u64(code, 0xAAAAAAAAAAAAAAAA)) 
        };
#else
        return { 
            static_cast<uint32_t>(compact_bits_2d(code)), 
            static_cast<uint32_t>(compact_bits_2d(code >> 1)) 
        };
#endif
    }

    /**
     * @brief Encodes 3D coordinates into a Morton code.
     */
    [[nodiscard]] inline uint64_t encode_3d(uint32_t x, uint32_t y, uint32_t z) {
#if defined(__BMI2__)
        return _pdep_u64(z, 0x4924924924924924) | 
               _pdep_u64(y, 0x2492492492492492) | 
               _pdep_u64(x, 0x9249249249249249);
#else
        return (spread_bits_3d(z) << 2) | (spread_bits_3d(y) << 1) | spread_bits_3d(x);
#endif
    }

    /**
     * @brief Decodes a Morton code into 3D coordinates.
     */
    [[nodiscard]] inline std::array<uint32_t, 3> decode_3d(uint64_t code) {
#if defined(__BMI2__)
        return { 
            static_cast<uint32_t>(_pext_u64(code, 0x9249249249249249)), 
            static_cast<uint32_t>(_pext_u64(code, 0x2492492492492492)), 
            static_cast<uint32_t>(_pext_u64(code, 0x4924924924924924)) 
        };
#else
        return { 
            static_cast<uint32_t>(compact_bits_3d(code)), 
            static_cast<uint32_t>(compact_bits_3d(code >> 1)), 
            static_cast<uint32_t>(compact_bits_3d(code >> 2)) 
        };
#endif
    }
} // namespace morton

/**
 * @brief Utilities for parallel algorithms (Prefix Sums).
 * Crucial for lock-free parallel refinement/coarsening.
 */
namespace parallel {
    template <typename T>
    void exclusive_scan(const std::vector<T>& in, std::vector<uint64_t>& out) {
        if (in.empty()) return;
        if (out.size() != in.size()) out.resize(in.size());
        // Uses parallel execution policy for scalability
        std::exclusive_scan(std::execution::par_unseq, in.begin(), in.end(), out.begin(), 0ULL);
    }
}

} // namespace amr