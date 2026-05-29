/**
 * @file core.hpp
 * @brief Core utilities for Distributed AMR.
 */
#pragma once

#include <mpi.h>
#include <cstdint>
#include <array>
#include <vector>
#include <algorithm>
#include <cmath>
#include <concepts>
#include <compare>
#include <omp.h>
#include <numbers>
#include <bit>
#include <type_traits>
#include <iostream>

#if defined(__x86_64__) || defined(_M_X64)
    #if defined(__BMI2__)
    #include <immintrin.h>
    #endif
#endif

namespace amr {

namespace constants {
    using std::numbers::sqrt3;
    using std::numbers::sqrt2;
    constexpr double SQRT3_OVER_2 = sqrt3 / 2.0;
    constexpr double SQRT2_OVER_2 = sqrt2 / 2.0;
}

struct MortonCode {
    uint64_t value;
    auto operator<=>(const MortonCode&) const = default;
    [[nodiscard]] constexpr MortonCode operator&(uint64_t mask) const { return {value & mask}; }
    [[nodiscard]] constexpr MortonCode operator|(uint64_t mask) const { return {value | mask}; }
    [[nodiscard]] constexpr MortonCode operator>>(int shift) const { return {value >> shift}; }
    [[nodiscard]] constexpr MortonCode operator<<(int shift) const { return {value << shift}; }
};

struct Coordinate {
    uint32_t value;
    auto operator<=>(const Coordinate&) const = default;
    constexpr Coordinate operator+(const Coordinate& other) const { return {value + other.value}; }
    constexpr Coordinate operator-(const Coordinate& other) const { return {value - other.value}; }
    [[nodiscard]] double to_double() const { return static_cast<double>(value); }
};

namespace morton {
    static constexpr uint64_t MASK2_X = 0x5555555555555555;
    static constexpr uint64_t MASK2_Y = 0xAAAAAAAAAAAAAAAA;
    static constexpr uint64_t MASK3_X = 0x9249249249249249;
    static constexpr uint64_t MASK3_Y = 0x2492492492492492;
    static constexpr uint64_t MASK3_Z = 0x4924924924924924;

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
    [[nodiscard]] constexpr uint32_t compact_bits_2d(uint64_t x) {
        x &= 0x5555555555555555;
        x = (x ^ (x >> 1)) & 0x3333333333333333;
        x = (x ^ (x >> 2)) & 0x0F0F0F0F0F0F0F0F;
        x = (x ^ (x >> 4)) & 0x00FF00FF00FF00FF;
        x = (x ^ (x >> 8)) & 0x0000FFFF0000FFFF;
        x = (x ^ (x >> 16)) & 0x00000000FFFFFFFF;
        return static_cast<uint32_t>(x);
    }
    [[nodiscard]] constexpr uint64_t spread_bits_3d(uint32_t a) {
        uint64_t x = static_cast<uint64_t>(a) & 0x1FFFFF;
        x = (x | (x << 32)) & 0x1F00000000FFFF;
        x = (x | (x << 16)) & 0x1F0000FF0000FF;
        x = (x | (x << 8))  & 0x100F00F00F00F00F;
        x = (x | (x << 4))  & 0x10C30C30C30C30C3;
        x = (x | (x << 2))  & 0x1249249249249249;
        return x;
    }
    [[nodiscard]] constexpr uint32_t compact_bits_3d(uint64_t x) {
        x &= 0x1249249249249249;
        x = (x ^ (x >> 2))  & 0x10C30C30C30C30C3;
        x = (x ^ (x >> 4))  & 0x100F00F00F00F00F;
        x = (x ^ (x >> 8))  & 0x1F0000FF0000FF;
        x = (x ^ (x >> 16)) & 0x1F00000000FFFF;
        x = (x ^ (x >> 32)) & 0x1FFFFF;
        return static_cast<uint32_t>(x);
    }
    [[nodiscard]] inline MortonCode encode_2d(Coordinate x, Coordinate y) {
#if defined(__BMI2__)
        return {_pdep_u64(x.value, MASK2_X) | _pdep_u64(y.value, MASK2_Y)};
#else
        return { spread_bits_2d(x.value) | (spread_bits_2d(y.value) << 1) };
#endif
    }
    [[nodiscard]] inline std::array<Coordinate, 2> decode_2d(MortonCode code) {
#if defined(__BMI2__)
        return { Coordinate{static_cast<uint32_t>(_pext_u64(code.value, MASK2_X))}, 
                 Coordinate{static_cast<uint32_t>(_pext_u64(code.value, MASK2_Y))} };
#else
        return { Coordinate{compact_bits_2d(code.value)}, Coordinate{compact_bits_2d(code.value >> 1)} };
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
        return { Coordinate{static_cast<uint32_t>(_pext_u64(code.value, MASK3_X))}, 
                 Coordinate{static_cast<uint32_t>(_pext_u64(code.value, MASK3_Y))}, 
                 Coordinate{static_cast<uint32_t>(_pext_u64(code.value, MASK3_Z))} };
#else
        return { Coordinate{compact_bits_3d(code.value)}, 
                 Coordinate{compact_bits_3d(code.value >> 1)}, 
                 Coordinate{compact_bits_3d(code.value >> 2)} };
#endif
    }
}

namespace parallel {
    template <typename T, typename U>
    void exclusive_scan(const std::vector<T>& in, std::vector<U>& out, std::vector<uint64_t>& workspace) {
        if (in.empty()) return;
        if (out.size() != in.size()) out.resize(in.size());
        
        int max_threads = omp_get_max_threads();
        if (workspace.size() < static_cast<size_t>(max_threads + 1)) workspace.resize(max_threads + 1);
        
        std::fill(workspace.begin(), workspace.end(), 0);

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
        
        workspace[0] = 0;
        for (int i = 1; i <= max_threads; ++i) workspace[i] += workspace[i - 1];
        
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
}

namespace mpi {
struct Context {
    int rank = 0; int size = 1;
    Context(int& argc, char**& argv) {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
    }
    ~Context() { MPI_Finalize(); }
};
inline bool all_reduce_or(bool local_val) {
    int local_int = local_val ? 1 : 0;
    int global_int = 0;
    MPI_Allreduce(&local_int, &global_int, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
    return global_int == 1;
}
template <typename T>
inline T all_reduce_sum(T local_val) {
    T global_val = 0;
    if constexpr (std::is_same_v<T, double>) {
        MPI_Allreduce(&local_val, &global_val, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    } else if constexpr (std::is_same_v<T, uint64_t>) {
        MPI_Allreduce(&local_val, &global_val, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    } else {
         MPI_Allreduce(&local_val, &global_val, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    }
    return global_val;
}
inline std::vector<uint64_t> build_partition_map(uint64_t local_first_code) {
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::vector<uint64_t> map(size);
    
    // 1. Gather starts (Empty ranks contribute UINT64_MAX)
    MPI_Allgather(&local_first_code, 1, MPI_UINT64_T, map.data(), 1, MPI_UINT64_T, MPI_COMM_WORLD);

    // 2. Post-process to fix unsorted MAX values
    // We scan backwards. If a rank is empty (MAX), it effectively starts 
    // where the next rank starts.
    uint64_t next_valid = std::numeric_limits<uint64_t>::max();
    for (int i = size - 1; i >= 0; --i) {
        if (map[i] == std::numeric_limits<uint64_t>::max()) {
            map[i] = next_valid;
        } else {
            next_valid = map[i];
        }
    }
    
    // Note: If the tail ranks are empty, they will remain MAX. 
    // This is acceptable as they lie beyond the last valid code.
    return map;
}
}
}