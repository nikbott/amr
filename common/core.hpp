/**
 * @file common/core.hpp
 * @brief Shared AMR core: Morton (Z-order) encoding, strong types, constants.
 *
 * Single source of truth for the bit-twiddling that every backend (omp/, mpi/,
 * cuda/) used to duplicate. Header-only and `constexpr`; functions that the GPU
 * backend runs on-device are tagged `AMR_HD` (== `__host__ __device__` under
 * nvcc, nothing otherwise). The BMI2 (PDEP/PEXT) fast path is host-x86 only and
 * is compiled out of the CUDA device pass.
 *
 * Backend-specific pieces stay in each backend's own core (OMP `Uninit` +
 * `parallel::exclusive_scan`, MPI's `mpi::` collectives, CUDA's Thrust glue).
 *
 * @cite Burstedde, Wilcox, Ghattas (2011) p4est. @cite Holke (2018) t8code.
 */
#pragma once

#include <array>
#include <cstdint>

#if defined(__CUDACC__)
#define AMR_HD __host__ __device__
#else
#define AMR_HD
#endif

// Host-side BMI2 acceleration on x86 only; never on the CUDA device pass.
#if !defined(__CUDA_ARCH__) && (defined(__x86_64__) || defined(_M_X64)) && defined(__BMI2__)
#include <immintrin.h>
#define AMR_HAS_BMI2 1
#endif

namespace amr {

// --- Geometric Constants ---
// Literals (not std::numbers) so the values are usable on the CUDA device path.
namespace constants {
constexpr double SQRT3_OVER_2 = 0.8660254037844386;
constexpr double SQRT2_OVER_2 = 0.7071067811865476;
}  // namespace constants

/**
 * @brief Strong type for the 64-bit Morton (Z-order) locational code.
 * Explicit comparison/bitwise ops (not `operator<=>`) for CUDA portability.
 */
struct MortonCode {
    uint64_t value;
    AMR_HD constexpr bool operator==(const MortonCode& o) const { return value == o.value; }
    AMR_HD constexpr bool operator!=(const MortonCode& o) const { return value != o.value; }
    AMR_HD constexpr bool operator<(const MortonCode& o) const { return value < o.value; }
    AMR_HD constexpr bool operator<=(const MortonCode& o) const { return value <= o.value; }
    AMR_HD constexpr bool operator>(const MortonCode& o) const { return value > o.value; }
    AMR_HD constexpr bool operator>=(const MortonCode& o) const { return value >= o.value; }
    [[nodiscard]] AMR_HD constexpr MortonCode operator&(uint64_t mask) const {
        return {value & mask};
    }
    [[nodiscard]] AMR_HD constexpr MortonCode operator|(uint64_t mask) const {
        return {value | mask};
    }
    [[nodiscard]] AMR_HD constexpr MortonCode operator>>(int shift) const {
        return {value >> shift};
    }
    [[nodiscard]] AMR_HD constexpr MortonCode operator<<(int shift) const {
        return {value << shift};
    }
};

/** @brief Strong type for an integer grid coordinate. */
struct Coordinate {
    uint32_t value;
    AMR_HD constexpr bool operator==(const Coordinate& o) const { return value == o.value; }
    AMR_HD constexpr bool operator!=(const Coordinate& o) const { return value != o.value; }
    AMR_HD constexpr bool operator<(const Coordinate& o) const { return value < o.value; }
    AMR_HD constexpr bool operator<=(const Coordinate& o) const { return value <= o.value; }
    AMR_HD constexpr bool operator>(const Coordinate& o) const { return value > o.value; }
    AMR_HD constexpr bool operator>=(const Coordinate& o) const { return value >= o.value; }
    [[nodiscard]] AMR_HD constexpr Coordinate operator+(const Coordinate& o) const {
        return {value + o.value};
    }
    [[nodiscard]] AMR_HD constexpr Coordinate operator-(const Coordinate& o) const {
        return {value - o.value};
    }
    [[nodiscard]] AMR_HD constexpr double to_double() const { return static_cast<double>(value); }
};

// ==================================================================================
// MORTON OPERATIONS
// ==================================================================================
namespace morton {

constexpr uint64_t MASK2_X = 0x5555555555555555;  // 0101...
constexpr uint64_t MASK2_Y = 0xAAAAAAAAAAAAAAAA;  // 1010...
constexpr uint64_t MASK3_X = 0x9249249249249249;  // 100100...
constexpr uint64_t MASK3_Y = 0x2492492492492492;  // 010010...
constexpr uint64_t MASK3_Z = 0x4924924924924924;  // 001001...

// --- SWAR bit spread/compact (no lookup tables, host + device) ---

[[nodiscard]] AMR_HD constexpr uint64_t spread_bits_2d(uint32_t a) {
    uint64_t x = static_cast<uint64_t>(a);
    x = (x | (x << 32)) & 0x00000000FFFFFFFF;
    x = (x | (x << 16)) & 0x0000FFFF0000FFFF;
    x = (x | (x << 8)) & 0x00FF00FF00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0F;
    x = (x | (x << 2)) & 0x3333333333333333;
    x = (x | (x << 1)) & 0x5555555555555555;
    return x;
}
[[nodiscard]] AMR_HD constexpr uint32_t compact_bits_2d(uint64_t x) {
    x &= 0x5555555555555555;
    x = (x ^ (x >> 1)) & 0x3333333333333333;
    x = (x ^ (x >> 2)) & 0x0F0F0F0F0F0F0F0F;
    x = (x ^ (x >> 4)) & 0x00FF00FF00FF00FF;
    x = (x ^ (x >> 8)) & 0x0000FFFF0000FFFF;
    x = (x ^ (x >> 16)) & 0x00000000FFFFFFFF;
    return static_cast<uint32_t>(x);
}
[[nodiscard]] AMR_HD constexpr uint64_t spread_bits_3d(uint32_t a) {
    uint64_t x = static_cast<uint64_t>(a) & 0x1FFFFF;
    x = (x | (x << 32)) & 0x1F00000000FFFF;
    x = (x | (x << 16)) & 0x1F0000FF0000FF;
    x = (x | (x << 8)) & 0x100F00F00F00F00F;
    x = (x | (x << 4)) & 0x10C30C30C30C30C3;
    x = (x | (x << 2)) & 0x1249249249249249;
    return x;
}
[[nodiscard]] AMR_HD constexpr uint32_t compact_bits_3d(uint64_t x) {
    x &= 0x1249249249249249;
    x = (x ^ (x >> 2)) & 0x10C30C30C30C30C3;
    x = (x ^ (x >> 4)) & 0x100F00F00F00F00F;
    x = (x ^ (x >> 8)) & 0x1F0000FF0000FF;
    x = (x ^ (x >> 16)) & 0x1F00000000FFFF;
    x = (x ^ (x >> 32)) & 0x1FFFFF;
    return static_cast<uint32_t>(x);
}

// --- encode/decode ---
// BMI2 on host x86; SWAR otherwise and on the device path.

[[nodiscard]] AMR_HD inline MortonCode encode_2d(Coordinate x, Coordinate y) {
#if defined(AMR_HAS_BMI2)
    return {_pdep_u64(x.value, MASK2_X) | _pdep_u64(y.value, MASK2_Y)};
#else
    return {spread_bits_2d(x.value) | (spread_bits_2d(y.value) << 1)};
#endif
}
[[nodiscard]] AMR_HD inline MortonCode encode_3d(Coordinate x, Coordinate y, Coordinate z) {
#if defined(AMR_HAS_BMI2)
    return {_pdep_u64(z.value, MASK3_Z) | _pdep_u64(y.value, MASK3_Y) |
            _pdep_u64(x.value, MASK3_X)};
#else
    return {spread_bits_3d(x.value) | (spread_bits_3d(y.value) << 1) |
            (spread_bits_3d(z.value) << 2)};
#endif
}

// Out-parameter form (device-friendly; used by the CUDA backend).
AMR_HD inline void decode_2d(MortonCode code, Coordinate& x, Coordinate& y) {
#if defined(AMR_HAS_BMI2)
    x.value = static_cast<uint32_t>(_pext_u64(code.value, MASK2_X));
    y.value = static_cast<uint32_t>(_pext_u64(code.value, MASK2_Y));
#else
    x.value = compact_bits_2d(code.value);
    y.value = compact_bits_2d(code.value >> 1);
#endif
}
AMR_HD inline void decode_3d(MortonCode code, Coordinate& x, Coordinate& y, Coordinate& z) {
#if defined(AMR_HAS_BMI2)
    x.value = static_cast<uint32_t>(_pext_u64(code.value, MASK3_X));
    y.value = static_cast<uint32_t>(_pext_u64(code.value, MASK3_Y));
    z.value = static_cast<uint32_t>(_pext_u64(code.value, MASK3_Z));
#else
    x.value = compact_bits_3d(code.value);
    y.value = compact_bits_3d(code.value >> 1);
    z.value = compact_bits_3d(code.value >> 2);
#endif
}

// Array-returning form (host only; used by the OMP/MPI backends).
[[nodiscard]] inline std::array<Coordinate, 2> decode_2d(MortonCode code) {
    Coordinate x, y;
    decode_2d(code, x, y);
    return {x, y};
}
[[nodiscard]] inline std::array<Coordinate, 3> decode_3d(MortonCode code) {
    Coordinate x, y, z;
    decode_3d(code, x, y, z);
    return {x, y, z};
}

/// Deepest level a tree may use: its Morton codes then stay below 2^63, so
/// UINT64_MAX (neighbor_code's "no neighbour") is never a cell's code. 21 in
/// 3D, 31 in 2D.
[[nodiscard]] AMR_HD constexpr int max_level_limit(int dim) {
    return dim == 3 ? 21 : 31;
}

/**
 * @brief Morton code of a cell's same-level neighbour.
 * @details The cell `code` at `level` moves dir[k] in {-1, 0, 1} cells along
 * each axis k, by integer arithmetic on the dilated coordinates, so no
 * decode/encode round trip is needed. Returns UINT64_MAX when the neighbour
 * leaves the domain [0, 2^max_level)^DIM, for any move from level 0 (a level-0
 * cell is the whole domain), and for a level or max_level out of range.
 */
template <int DIM>
[[nodiscard]] AMR_HD inline uint64_t neighbor_code(uint64_t code,
                                                   int level,
                                                   int max_level,
                                                   const int* dir) {
    constexpr uint64_t kNone = ~uint64_t{0};
    constexpr uint64_t kAxis = DIM == 3 ? MASK3_X : MASK2_X;  // axis k's bits: kAxis << k
    if (level < 0 || level > max_level || max_level > max_level_limit(DIM))
        return kNone;
    const unsigned shift = static_cast<unsigned>(max_level - level) * DIM;
    const uint64_t domain = (uint64_t{1} << (DIM * max_level)) - 1;  // DIM * max_level <= 63
    uint64_t c = code;
    for (int k = 0; k < DIM; ++k) {
        if (dir[k] == 0)
            continue;
        if (level == 0)
            return kNone;
        const uint64_t mask = kAxis << k;
        const uint64_t one = uint64_t{1} << (shift + static_cast<unsigned>(k));
        if (dir[k] > 0) {
            // Out of the domain iff this axis already sits at its largest
            // coordinate aligned to this level.
            if ((c & mask) == ((mask & domain) & ~(one - 1)))
                return kNone;
            c = (((c | ~mask) + one) & mask) | (c & ~mask);
        } else {
            if ((c & mask) < one)
                return kNone;
            c = (((c & mask) - one) & mask) | (c & ~mask);
        }
    }
    return c;
}

}  // namespace morton
}  // namespace amr
