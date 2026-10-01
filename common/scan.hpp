/**
 * @file scan.hpp
 * @brief The OpenMP exclusive prefix sum the omp and mpi backends share.
 */
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include <omp.h>

namespace amr::parallel {

/**
 * @brief out[i] = in[0] + ... + in[i-1], in parallel.
 *
 * @details Each thread sums its chunk, one thread prefix-sums the chunk totals,
 * then each thread writes its chunk's running sum. All three phases run in one
 * parallel region, so they see the same team: two regions can be given teams of
 * different sizes (nested parallelism under a thread limit), which mismatches
 * the chunks and reads stale totals.
 *
 * @param workspace scratch for the chunk totals, reused across calls
 */
template <typename T, typename U>
void exclusive_scan(const std::vector<T>& in,
                    std::vector<U>& out,
                    std::vector<uint64_t>& workspace) {
    if (in.empty())
        return;
    out.resize(in.size());
    const std::size_t n = in.size();
#pragma omp parallel
    {
        const auto n_threads = static_cast<std::size_t>(omp_get_num_threads());
        const auto tid = static_cast<std::size_t>(omp_get_thread_num());
#pragma omp single
        workspace.assign(n_threads + 1, 0);  // implied barrier
        const std::size_t chunk = (n + n_threads - 1) / n_threads;
        const std::size_t start = std::min(n, tid * chunk);
        const std::size_t end = std::min(n, start + chunk);
        uint64_t sum = 0;
        for (std::size_t i = start; i < end; ++i)
            sum += in[i];
        workspace[tid + 1] = sum;
#pragma omp barrier
#pragma omp single
        for (std::size_t t = 1; t <= n_threads; ++t)
            workspace[t] += workspace[t - 1];  // implied barrier
        uint64_t running = workspace[tid];
        for (std::size_t i = start; i < end; ++i) {
            out[i] = running;
            running += in[i];
        }
    }
}

}  // namespace amr::parallel
