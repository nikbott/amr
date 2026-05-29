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

#include "../common/core.hpp"  // amr::MortonCode, Coordinate, morton::, constants

namespace amr {

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

// MortonCode, Coordinate, the morton:: namespace, and constants now live in
// common/core.hpp (shared with the mpi/ and cuda/ backends).

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