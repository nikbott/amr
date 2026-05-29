/**
 * @file core.hpp
 * @brief Core utilities for AMR: Morton Codes, Strong Types, and Parallel Primitives.
 * * @details 
 * Implements the bijection between Cartesian coordinates and Morton indices (Z-order curve).
 * This encoding allows the "Linear Tree" approach where the grid is represented as a 
 * sorted array of leaf nodes rather than a pointer-based tree structure.
 * * Key Algorithms:
 * - **Bit Interleaving**: Maps multidimensional coordinates to a 1D index.
 * - **BMI2 Acceleration**: Uses Intel PDEP/PEXT instructions for O(1) encoding/decoding.
 * - **SWAR Fallback**: SIMD Within A Register techniques for non-BMI2 hardware.
 * * @cite Burstedde, C., Wilcox, L. C., & Ghattas, O. (2011). p4est: Scalable algorithms for parallel 
 * adaptive mesh refinement on forests of octrees. SIAM Journal on Scientific Computing.
 * @cite Holke, J. (2018). Scalable Algorithms for Parallel Tree-based Adaptive Mesh Refinement.
 */

#pragma once

#include <cstdint>
#include <array>
#include <vector>
#include <algorithm>
#include <cmath>
#include <concepts>
#include <compare>
#include <omp.h>
#include <numbers> // C++20
#include <bit>
#include <type_traits>

#if defined(__x86_64__) || defined(_M_X64)
    #if defined(__BMI2__)
    #include <immintrin.h>
    #endif
#endif

namespace amr {

// --- Geometric Constants ---
namespace constants {
    using std::numbers::sqrt3;
    using std::numbers::sqrt2;
    constexpr double SQRT3_OVER_2 = sqrt3 / 2.0;
    constexpr double SQRT2_OVER_2 = sqrt2 / 2.0;
}

/**
 * @brief Wrapper to skip initialization of POD types in containers.
 * @details 
 * In high-performance AMR, resizing vectors (e.g., for `leaf_codes`) is frequent.
 * Standard `std::vector::resize` zero-initializes memory, which is redundant when 
 * followed by a parallel write. This wrapper prevents that overhead.
 * * @tparam T The trivially destructible type to wrap.
 */
template <typename T>
struct Uninit {
    static_assert(std::is_trivially_destructible_v<T>, "Uninit is only safe for trivially destructible types.");
    T value;
    
    Uninit() noexcept {} 
    constexpr Uninit(T v) noexcept : value(v) {}
    
    constexpr operator T&() noexcept { return value; }
    constexpr operator const T&() const noexcept { return value; }
};

/**
 * @brief Strong type for the 64-bit Morton Index.
 * @details Represents a locational code on the Z-order curve.
 */
struct MortonCode {
    uint64_t value;
    auto operator<=>(const MortonCode&) const = default;
    
    // Bitwise operators for neighbor finding logic
    [[nodiscard]] constexpr MortonCode operator&(uint64_t mask) const { return {value & mask}; }
    [[nodiscard]] constexpr MortonCode operator|(uint64_t mask) const { return {value | mask}; }
    [[nodiscard]] constexpr MortonCode operator>>(int shift) const { return {value >> shift}; }
    [[nodiscard]] constexpr MortonCode operator<<(int shift) const { return {value << shift}; }
};

/**
 * @brief Strong type for integer coordinates.
 */
struct Coordinate {
    uint32_t value;
    auto operator<=>(const Coordinate&) const = default;
    constexpr Coordinate operator+(const Coordinate& other) const { return {value + other.value}; }
    constexpr Coordinate operator-(const Coordinate& other) const { return {value - other.value}; }
    [[nodiscard]] double to_double() const { return static_cast<double>(value); }
};

// ==================================================================================
// MORTON OPERATIONS (Encoding, Decoding, Arithmetic)
// ==================================================================================
namespace morton {

    static constexpr uint64_t MASK2_X = 0x5555555555555555; // 0101...
    static constexpr uint64_t MASK2_Y = 0xAAAAAAAAAAAAAAAA; // 1010...

    static constexpr uint64_t MASK3_X = 0x9249249249249249; // 100100...
    static constexpr uint64_t MASK3_Y = 0x2492492492492492; // 010010...
    static constexpr uint64_t MASK3_Z = 0x4924924924924924; // 001001...

    // --- SWAR Primitives (No Lookup Tables) ---

    // Spread bits for 2D (0000dcba -> 0d0c0b0a)
    [[nodiscard]] constexpr uint64_t spread_bits_2d(uint32_t a) {
        uint64_t x = static_cast<uint64_t>(a);
        x = (x | (x << 32)) & 0x00000000FFFFFFFF;
        x = (x | (x << 16)) & 0x0000FFFF0000FFFF;
        x = (x | (x << 8))  & 0x00FF00FF00FF00FF;
        x = (x | (x << 4))  & 0x0F0F0F0F0F0F0F0F;
        x = (x | (x << 2))  & 0x3333333333333333;
        x = (x | (x << 1))  & 0x5555555555555555;
        return x;
    }

    // Compact bits for 2D (0d0c0b0a -> 0000dcba)
    [[nodiscard]] constexpr uint32_t compact_bits_2d(uint64_t x) {
        x &= 0x5555555555555555;
        x = (x ^ (x >> 1)) & 0x3333333333333333;
        x = (x ^ (x >> 2)) & 0x0F0F0F0F0F0F0F0F;
        x = (x ^ (x >> 4)) & 0x00FF00FF00FF00FF;
        x = (x ^ (x >> 8)) & 0x0000FFFF0000FFFF;
        x = (x ^ (x >> 16)) & 0x00000000FFFFFFFF;
        return static_cast<uint32_t>(x);
    }

    // Spread bits for 3D (00000cba -> 00c00b00a)
    [[nodiscard]] constexpr uint64_t spread_bits_3d(uint32_t a) {
        uint64_t x = static_cast<uint64_t>(a) & 0x1FFFFF;
        x = (x | (x << 32)) & 0x1F00000000FFFF;
        x = (x | (x << 16)) & 0x1F0000FF0000FF;
        x = (x | (x << 8))  & 0x100F00F00F00F00F;
        x = (x | (x << 4))  & 0x10C30C30C30C30C3;
        x = (x | (x << 2))  & 0x1249249249249249;
        return x;
    }

    // Compact bits for 3D (00c00b00a -> 00000cba)
    [[nodiscard]] constexpr uint32_t compact_bits_3d(uint64_t x) {
        x &= 0x1249249249249249;
        x = (x ^ (x >> 2))  & 0x10C30C30C30C30C3;
        x = (x ^ (x >> 4))  & 0x100F00F00F00F00F;
        x = (x ^ (x >> 8))  & 0x1F0000FF0000FF;
        x = (x ^ (x >> 16)) & 0x1F00000000FFFF;
        x = (x ^ (x >> 32)) & 0x1FFFFF;
        return static_cast<uint32_t>(x);
    }

    // --- API ---

    [[nodiscard]] inline MortonCode encode_2d(Coordinate x, Coordinate y) {
#if defined(__BMI2__)
        return {_pdep_u64(x.value, MASK2_X) | _pdep_u64(y.value, MASK2_Y)};
#else
        return { spread_bits_2d(x.value) | (spread_bits_2d(y.value) << 1) };
#endif
    }

    [[nodiscard]] inline std::array<Coordinate, 2> decode_2d(MortonCode code) {
#if defined(__BMI2__)
        return { 
            Coordinate{static_cast<uint32_t>(_pext_u64(code.value, MASK2_X))}, 
            Coordinate{static_cast<uint32_t>(_pext_u64(code.value, MASK2_Y))} 
        };
#else
        return { 
            Coordinate{compact_bits_2d(code.value)}, 
            Coordinate{compact_bits_2d(code.value >> 1)} 
        };
#endif
    }

    [[nodiscard]] inline MortonCode encode_3d(Coordinate x, Coordinate y, Coordinate z) {
#if defined(__BMI2__)
        return {_pdep_u64(z.value, MASK3_Z) | _pdep_u64(y.value, MASK3_Y) | _pdep_u64(x.value, MASK3_X)};
#else
        return { spread_bits_3d(x.value) | (spread_bits_3d(y.value) << 1) | (spread_bits_3d(z.value) << 2) };
#endif
    }

    [[nodiscard]] inline std::array<Coordinate, 3> decode_3d(MortonCode code) {
#if defined(__BMI2__)
        return { 
            Coordinate{static_cast<uint32_t>(_pext_u64(code.value, MASK3_X))}, 
            Coordinate{static_cast<uint32_t>(_pext_u64(code.value, MASK3_Y))}, 
            Coordinate{static_cast<uint32_t>(_pext_u64(code.value, MASK3_Z))} 
        };
#else
        return { 
            Coordinate{compact_bits_3d(code.value)}, 
            Coordinate{compact_bits_3d(code.value >> 1)}, 
            Coordinate{compact_bits_3d(code.value >> 2)} 
        };
#endif
    }
} // namespace morton

// ==================================================================================
// PARALLEL PRIMITIVES
// ==================================================================================
namespace parallel {
    /**
     * @brief Parallel exclusive prefix sum (Scan).
     * @details Computes offsets for variable-output parallel operations (e.g. Refine).
     * Uses OMP to perform thread-local sums followed by a serial offset fixup.
     */
    template <typename T, typename U>
    void exclusive_scan(const std::vector<T>& in, std::vector<U>& out, std::vector<uint64_t>& workspace) {
        if (in.empty()) return;
        if (out.size() != in.size()) out.resize(in.size());

        int max_threads = omp_get_max_threads();
        if (workspace.size() < static_cast<size_t>(max_threads + 1)) {
            workspace.resize(max_threads + 1);
        }

        // Phase 1: Local Sums
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int n_threads = omp_get_num_threads();
            size_t n = in.size();
            size_t chunk = (n + n_threads - 1) / n_threads;
            size_t start = tid * chunk;
            size_t end = std::min(start + chunk, n);

            uint64_t local_sum = 0;
            for (size_t i = start; i < end; ++i) local_sum += in[i];
            workspace[tid + 1] = local_sum;
        }

        // Phase 2: Prefix sum of block totals (Serial)
        workspace[0] = 0;
        for (int i = 1; i <= max_threads; ++i) workspace[i] += workspace[i - 1];

        // Phase 3: Local Prefix Sums
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int n_threads = omp_get_num_threads();
            size_t n = in.size();
            size_t chunk = (n + n_threads - 1) / n_threads;
            size_t start = tid * chunk;
            size_t end = std::min(start + chunk, n);

            uint64_t running = workspace[tid]; 
            for (size_t i = start; i < end; ++i) {
                out[i] = running;
                running += in[i];
            }
        }
    }
} // namespace parallel
} // namespace amr