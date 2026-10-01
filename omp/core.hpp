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
 * * @cite Burstedde, C., Wilcox, L. C., & Ghattas, O. (2011). p4est: Scalable algorithms for
 * parallel adaptive mesh refinement on forests of octrees. SIAM Journal on Scientific Computing.
 * @cite Holke, J. (2018). Scalable Algorithms for Parallel Tree-based Adaptive Mesh Refinement.
 */

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <compare>
#include <concepts>
#include <cstdint>
#include <numbers>  // C++20
#include <type_traits>
#include <vector>

#include <omp.h>

#include "../common/core.hpp"  // amr::MortonCode, Coordinate, morton::, constants
#include "../common/scan.hpp"  // amr::parallel::exclusive_scan

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
    static_assert(std::is_trivially_destructible_v<T>,
                  "Uninit is only safe for trivially destructible types.");
    T value;

    Uninit() noexcept {}
    constexpr Uninit(T v) noexcept : value(v) {}

    constexpr operator T&() noexcept { return value; }
    constexpr operator const T&() const noexcept { return value; }
};

// MortonCode, Coordinate, the morton:: namespace, and constants now live in
// common/core.hpp (shared with the mpi/ and cuda/ backends).

}  // namespace amr
