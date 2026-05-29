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

#include "../common/core.hpp"  // amr::MortonCode, Coordinate, morton::, constants

namespace amr {

// MortonCode, Coordinate, the morton:: namespace, and constants now live in
// common/core.hpp (shared with the omp/ and cuda/ backends).

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