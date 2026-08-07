/**
 * @file tree.cuh
 * @brief CUDA Linear Quadtree/Octree Implementation.
 * @details Implements p4est/Holke algorithms.
 * FIXED: Uses Manual Double Buffering to avoid malloc-in-loop (Compatible with ALL CUDA versions).
 */
#pragma once

#include <cmath>
#include <stdexcept>
#include <string>

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

#include "core.cuh"

namespace amr {

// --- CUDA error checking -----------------------------------------------------
// Wrap any cudaXxx() API call. Throws std::runtime_error with the call text
// and human-readable error string. Asynchronous kernel errors surface here
// when paired with a subsequent cudaDeviceSynchronize() (already used below).
// CHECK_CUDA_LAUNCH() catches immediate launch-config errors (invalid grid
// dims, no kernel found, etc.); pair with CHECK_CUDA(cudaDeviceSynchronize())
// to catch runtime kernel errors too.
#define CHECK_CUDA(call)                                                             \
    do {                                                                             \
        cudaError_t cuda_err__ = (call);                                             \
        if (cuda_err__ != cudaSuccess) {                                             \
            throw std::runtime_error(std::string("CUDA error: ") +                   \
                                     cudaGetErrorString(cuda_err__) + " at " #call); \
        }                                                                            \
    } while (0)

#define CHECK_CUDA_LAUNCH()                                                       \
    do {                                                                          \
        cudaError_t launch_err__ = cudaGetLastError();                            \
        if (launch_err__ != cudaSuccess) {                                        \
            throw std::runtime_error(std::string("CUDA kernel launch failed: ") + \
                                     cudaGetErrorString(launch_err__));           \
        }                                                                         \
    } while (0)

// --- Helper Functions ---

template <typename KernelFunc>
void get_launch_config(KernelFunc kernel, int n, int& grid, int& block) {
    int minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &block, kernel, 0, 0);
    if (n == 0)
        grid = 0;
    else
        grid = (n + block - 1) / block;
}

// --- Helper Kernels ---

template <typename T>
HOST_DEVICE int device_lower_bound(const T* data, int n, T val) {
    int l = 0, r = n;
    while (l < r) {
        int mid = l + (r - l) / 2;
        // Optimization: Use __ldg for read-only cache
        if (__ldg(&data[mid]) < val)
            l = mid + 1;
        else
            r = mid;
    }
    return l;
}

template <int DIM>
HOST_DEVICE uint64_t get_neighbor_code(uint64_t code, int level, int max_level, const int* dir) {
    uint64_t mask_x, mask_y, mask_z;
    if (DIM == 3) {
        mask_x = morton::MASK3_X;
        mask_y = morton::MASK3_Y;
        mask_z = morton::MASK3_Z;
    } else {
        mask_x = morton::MASK2_X;
        mask_y = morton::MASK2_Y;
        mask_z = 0;
    }

    auto add_dim = [&](uint64_t c, uint64_t mask, int d) -> uint64_t {
        if (c == UINT64_MAX)
            return UINT64_MAX;
        if (d == 0)
            return c;

        uint64_t shift = max_level - level;
        uint64_t one;
        if (DIM == 2)
            one = 1ULL << (shift * 2);
        else
            one = 1ULL << (shift * 3);

        if (mask == mask_y)
            one <<= 1;
        if (mask == mask_z)
            one <<= 2;

        if (d > 0) {
            if ((c | ~mask) == UINT64_MAX)
                return UINT64_MAX;
            uint64_t sum = (c | ~mask) + one;
            return (sum & mask) | (c & ~mask);
        } else {
            if ((c & mask) < one)
                return UINT64_MAX;
            uint64_t diff = (c & mask) - one;
            return (diff & mask) | (c & ~mask);
        }
    };

    uint64_t next = code;
    next = add_dim(next, mask_x, dir[0]);
    next = add_dim(next, mask_y, dir[1]);
    if (DIM == 3)
        next = add_dim(next, mask_z, dir[2]);
    return next;
}

// --- Refinement Kernels ---

template <typename Oracle>
__global__ void k_mark_refine(
    const uint64_t* codes, const uint8_t* levels, int n, int* counts, Oracle oracle, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    if (oracle(MortonCode{codes[idx]}, levels[idx]))
        counts[idx] = (1 << dim);
    else
        counts[idx] = 1;
}

__global__ void k_scatter_refine(const uint64_t* old_codes,
                                 const uint8_t* old_levels,
                                 int n,
                                 const int* offsets,
                                 const int* counts,
                                 uint64_t* new_codes,
                                 uint8_t* new_levels,
                                 int max_level,
                                 int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    int pos = offsets[idx];
    int cnt = counts[idx];
    uint64_t code = old_codes[idx];
    int lvl = old_levels[idx];

    if (cnt == 1) {
        new_codes[pos] = code;
        new_levels[pos] = lvl;
    } else {
        int new_lvl = lvl + 1;
        uint64_t shift = (uint64_t)(max_level - new_lvl) * dim;
        for (int k = 0; k < cnt; ++k) {
            new_codes[pos + k] = code | ((uint64_t)k << shift);
            new_levels[pos + k] = new_lvl;
        }
    }
}

// --- Coarsening Kernels ---

template <typename Oracle>
__global__ void k_mark_coarsen(const uint64_t* codes,
                               const uint8_t* levels,
                               int n,
                               int* flags,
                               Oracle oracle,
                               int dim,
                               int max_level) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    flags[idx] = 1;  // Default keep

    int siblings = (1 << dim);
    if (idx + siblings > n)
        return;

    uint64_t code = codes[idx];
    int lvl = levels[idx];
    if (lvl == 0)
        return;

    uint64_t shift = (uint64_t)(max_level - lvl) * dim;
    uint64_t child_idx_in_parent = (code >> shift) & ((1ULL << dim) - 1);

    if (child_idx_in_parent != 0)
        return;

    bool family_intact = true;
    for (int k = 1; k < siblings; ++k) {
        if (levels[idx + k] != lvl) {
            family_intact = false;
            break;
        }
    }

    if (family_intact) {
        if (!oracle(MortonCode{code}, lvl - 1)) {
            flags[idx] = 2;  // Coarsen
            for (int k = 1; k < siblings; ++k)
                flags[idx + k] = 0;  // Discard
        }
    }
}

__global__ void k_scatter_coarsen(const uint64_t* old_codes,
                                  const uint8_t* old_levels,
                                  int n,
                                  const int* offsets,
                                  const int* flags,
                                  uint64_t* new_codes,
                                  uint8_t* new_levels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    int flag = flags[idx];
    if (flag == 0)
        return;

    int pos = offsets[idx];
    if (flag == 2) {
        new_codes[pos] = old_codes[idx];
        new_levels[pos] = old_levels[idx] - 1;
    } else {
        new_codes[pos] = old_codes[idx];
        new_levels[pos] = old_levels[idx];
    }
}

// --- Balance Kernel ---

template <int DIM>
__global__ void k_check_balance(const uint64_t* codes,
                                const uint8_t* levels,
                                int n,
                                int max_level,
                                int* flags,
                                int* violation_occured) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    uint64_t my_code = codes[idx];
    int my_lvl = levels[idx];

    int dirs[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    int num_dirs = (DIM == 2) ? 4 : 6;

    for (int d = 0; d < num_dirs; ++d) {
        uint64_t n_code = get_neighbor_code<DIM>(my_code, my_lvl, max_level, dirs[d]);
        if (n_code == UINT64_MAX)
            continue;

        int search_lvl = my_lvl - 2;
        if (search_lvl < 0)
            continue;

        uint64_t shift = (uint64_t)(max_level - search_lvl) * DIM;
        uint64_t mask = (shift >= 64) ? 0 : (~0ULL << shift);
        uint64_t target = n_code & mask;

        int found_idx = device_lower_bound(codes, n, target);

        bool violation = false;
        int violator_idx = -1;

        if (found_idx < n && codes[found_idx] == target) {
            if (levels[found_idx] <= search_lvl) {
                violation = true;
                violator_idx = found_idx;
            }
        } else if (found_idx > 0) {
            int prev = found_idx - 1;
            uint64_t prev_c = codes[prev];
            int prev_l = levels[prev];
            uint64_t size = 1ULL << ((max_level - prev_l) * DIM);
            if (prev_c <= target && (prev_c + size) > target) {
                if (prev_l <= search_lvl) {
                    violation = true;
                    violator_idx = prev;
                }
            }
        }

        if (violation) {
            flags[violator_idx] = 1;
            *violation_occured = 1;
        }
    }
}

// --- Active-Front Balance Kernels --------------------------------------------
// The classic balance() loop refines violators ONE level per pass, but re-checks
// the ENTIRE (growing) leaf array every pass. After the first pass, however, a
// cell can only NEWLY violate the 2:1 rule if one of its face-neighbours just
// got finer. So passes 2..K only need to inspect the advancing refinement front,
// not all N leaves -- the dominant cost on deep ripples (e.g. a fine crack in a
// coarse domain: 10 passes over ~40M cells).
//
// We track a `dirty` mask. Pass 1: all dirty. After refining cells {j} in a
// pass, the only cells that can newly violate are (a) the children of {j} and
// (b) the face-neighbours of {j} (a pre-existing fine cell adjacent to j may now
// need to re-flag j's still-too-coarse child). Marking both is provably a
// complete superset, so the produced flags -- and thus the final tree -- are
// byte-identical to balance(). See run_test_active_parity in tests.cu.

// Dirty-gated variant of k_check_balance: only leaves with dirty[idx] do work.
template <int DIM>
__global__ void k_check_balance_active(const uint64_t* codes,
                                       const uint8_t* levels,
                                       int n,
                                       int max_level,
                                       const uint8_t* dirty,
                                       int* flags,
                                       int* violation_occured) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    if (!dirty[idx])
        return;

    uint64_t my_code = codes[idx];
    int my_lvl = levels[idx];

    int dirs[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    int num_dirs = (DIM == 2) ? 4 : 6;

    for (int d = 0; d < num_dirs; ++d) {
        uint64_t n_code = get_neighbor_code<DIM>(my_code, my_lvl, max_level, dirs[d]);
        if (n_code == UINT64_MAX)
            continue;

        int search_lvl = my_lvl - 2;
        if (search_lvl < 0)
            continue;

        uint64_t shift = (uint64_t)(max_level - search_lvl) * DIM;
        uint64_t mask = (shift >= 64) ? 0 : (~0ULL << shift);
        uint64_t target = n_code & mask;

        int found_idx = device_lower_bound(codes, n, target);

        if (found_idx < n && codes[found_idx] == target) {
            if (levels[found_idx] <= search_lvl) {
                flags[found_idx] = 1;
                *violation_occured = 1;
            }
        } else if (found_idx > 0) {
            int prev = found_idx - 1;
            uint64_t prev_c = codes[prev];
            int prev_l = levels[prev];
            uint64_t size = 1ULL << ((uint64_t)(max_level - prev_l) * DIM);
            if (prev_c <= target && (prev_c + size) > target) {
                if (prev_l <= search_lvl) {
                    flags[prev] = 1;
                    *violation_occured = 1;
                }
            }
        }
    }
}

// Pass A: mark the just-created children dirty. Run over OLD cells gated on
// flags[i]==1; each refined cell's children occupy offsets[i] .. +2^DIM in the
// new array. The resulting mask doubles as the read-only "is a new child" gate.
__global__ void k_mark_children(
    int n, const int* flags, const int* offsets, uint8_t* child_mask, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    if (!flags[idx])
        return;
    int base = offsets[idx];
    int children = 1 << dim;
    for (int k = 0; k < children; ++k)
        child_mask[base + k] = 1;
}

// Pass B: from each NEW child, mark its equal-or-finer face-neighbours dirty.
// Seeding from the fine (child) side locates every distinct adjacent leaf that
// could flag the child next pass. A face's neighbour region spans the Morton
// range [n_code, n_code + child_cell_size); every leaf in that range is an
// equal-or-finer face neighbour, so we mark the whole range (not just the first
// -- a coarse cell can have many fine neighbours across one face). The coarser
// side needs no marking: the dirty child itself will flag it. dirty_out starts
// as a copy of child_mask and is a separate array from it -> no read/write race.
template <int DIM>
__global__ void k_mark_child_neighbors(const uint64_t* codes,
                                       const uint8_t* levels,
                                       int n,
                                       int max_level,
                                       const uint8_t* child_mask,
                                       uint8_t* dirty_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    if (!child_mask[idx])
        return;

    uint64_t my_code = codes[idx];
    int my_lvl = levels[idx];
    uint64_t my_size = 1ULL << ((uint64_t)(max_level - my_lvl) * DIM);

    int dirs[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    int num_dirs = (DIM == 2) ? 4 : 6;

    for (int d = 0; d < num_dirs; ++d) {
        uint64_t n_code = get_neighbor_code<DIM>(my_code, my_lvl, max_level, dirs[d]);
        if (n_code == UINT64_MAX)
            continue;

        // Mark every leaf whose code lies in [n_code, n_code + my_size): the
        // equal-or-finer neighbours sharing this face.
        int lo = device_lower_bound(codes, n, n_code);
        int hi = device_lower_bound(codes, n, n_code + my_size);
        for (int m = lo; m < hi; ++m)
            dirty_out[m] = 1;
    }
}

// --- Linear Tree Class ---

template <int DIM>
class LinearTree {
public:
    static constexpr int dim = DIM;
    int max_level;

    // Primary State
    thrust::device_vector<uint64_t> codes;
    thrust::device_vector<uint8_t> levels;

    // Scratchpad (Manual Memory Pool)
    // We keep these persistent to avoid reallocation in loops
    thrust::device_vector<uint64_t> scratch_codes;
    thrust::device_vector<uint8_t> scratch_levels;

    // Reused auxiliary buffers
    thrust::device_vector<int> aux_counts;
    thrust::device_vector<int> aux_offsets;
    thrust::device_vector<int> aux_flags;

    explicit LinearTree(int max_lvl) : max_level(max_lvl) {
        validate_config();

        codes.push_back(0);
        levels.push_back(0);

        // Reserve initial capacity to minimize early resizes
        size_t cap = 10000;
        codes.reserve(cap);
        levels.reserve(cap);
        scratch_codes.reserve(cap);
        scratch_levels.reserve(cap);
        aux_counts.reserve(cap);
        aux_offsets.reserve(cap);
        aux_flags.reserve(cap);
    }

    void validate_config() const {
        if constexpr (DIM == 3) {
            if (max_level > 21)
                throw std::runtime_error(
                    "Safety Violation: 3D Max Level > 21 is unsafe for 64-bit Morton codes.");
        } else {
            if (max_level > 32)
                throw std::runtime_error("Safety Violation: 2D Max Level > 32 is unsafe.");
        }
    }

    size_t size() const { return codes.size(); }
    uint64_t domain_width() const { return 1ULL << max_level; }

    void verify() const {
        thrust::host_vector<uint64_t> h_codes = codes;
        thrust::host_vector<uint8_t> h_levels = levels;
        if (h_codes.empty())
            throw std::runtime_error("Empty tree");
        if (!std::is_sorted(h_codes.begin(), h_codes.end()))
            throw std::runtime_error("Not sorted");
        // Simple volume check
        double total_vol = 0.0;
        uint64_t last_end = 0;
        for (size_t i = 0; i < h_codes.size(); ++i) {
            uint64_t c = h_codes[i];
            int l = h_levels[i];
            if (c < last_end)
                throw std::runtime_error("Overlap detected");
            double side = 1.0 / (double)(1ULL << l);
            total_vol += std::pow(side, DIM);
            last_end = c + (1ULL << (DIM * (max_level - l)));
        }
        if (std::abs(total_vol - 1.0) > 1e-9)
            throw std::runtime_error("Volume != 1.0");
    }

    template <typename Oracle>
    bool refine(Oracle oracle) {
        int n = codes.size();

        // Resize auxiliary buffers (reuse capacity if possible)
        aux_counts.resize(n);
        aux_offsets.resize(n);

        int grid, block;
        get_launch_config(k_mark_refine<Oracle>, n, grid, block);

        k_mark_refine<<<grid, block>>>(thrust::raw_pointer_cast(codes.data()),
                                       thrust::raw_pointer_cast(levels.data()),
                                       n,
                                       thrust::raw_pointer_cast(aux_counts.data()),
                                       oracle,
                                       DIM);
        CHECK_CUDA(cudaDeviceSynchronize());

        thrust::exclusive_scan(aux_counts.begin(), aux_counts.end(), aux_offsets.begin());
        int total = aux_offsets.back() + aux_counts.back();

        if (total == n)
            return false;

        // Write to scratch buffers instead of new local vectors
        scratch_codes.resize(total);
        scratch_levels.resize(total);

        get_launch_config(k_scatter_refine, n, grid, block);

        k_scatter_refine<<<grid, block>>>(thrust::raw_pointer_cast(codes.data()),
                                          thrust::raw_pointer_cast(levels.data()),
                                          n,
                                          thrust::raw_pointer_cast(aux_offsets.data()),
                                          thrust::raw_pointer_cast(aux_counts.data()),
                                          thrust::raw_pointer_cast(scratch_codes.data()),
                                          thrust::raw_pointer_cast(scratch_levels.data()),
                                          max_level,
                                          DIM);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Fast pointer swap: "codes" now owns the new data, "scratch" owns the old (to be reused)
        codes.swap(scratch_codes);
        levels.swap(scratch_levels);
        return true;
    }

    template <typename Oracle>
    bool coarsen(Oracle oracle) {
        int n = codes.size();
        aux_flags.resize(n);
        aux_offsets.resize(n);

        int grid, block;
        get_launch_config(k_mark_coarsen<Oracle>, n, grid, block);

        k_mark_coarsen<<<grid, block>>>(thrust::raw_pointer_cast(codes.data()),
                                        thrust::raw_pointer_cast(levels.data()),
                                        n,
                                        thrust::raw_pointer_cast(aux_flags.data()),
                                        oracle,
                                        DIM,
                                        max_level);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Use aux_counts as temporary mask storage
        aux_counts.resize(n);
        thrust::transform(
            aux_flags.begin(), aux_flags.end(), aux_counts.begin(), [] __device__(int f) {
                return f > 0 ? 1 : 0;
            });

        thrust::exclusive_scan(aux_counts.begin(), aux_counts.end(), aux_offsets.begin());
        int total = aux_offsets.back() + aux_counts.back();

        if (total == n)
            return false;

        scratch_codes.resize(total);
        scratch_levels.resize(total);

        get_launch_config(k_scatter_coarsen, n, grid, block);

        k_scatter_coarsen<<<grid, block>>>(thrust::raw_pointer_cast(codes.data()),
                                           thrust::raw_pointer_cast(levels.data()),
                                           n,
                                           thrust::raw_pointer_cast(aux_offsets.data()),
                                           thrust::raw_pointer_cast(aux_flags.data()),
                                           thrust::raw_pointer_cast(scratch_codes.data()),
                                           thrust::raw_pointer_cast(scratch_levels.data()));
        CHECK_CUDA(cudaDeviceSynchronize());

        codes.swap(scratch_codes);
        levels.swap(scratch_levels);
        return true;
    }

    // Number of ripple passes the last balance*() call performed.
    int last_balance_iters = 0;

    // Active-front 2:1 balance (production). Byte-identical output to balance_ref()
    // but passes 2..K only inspect the advancing refinement front (see the
    // k_check_balance_active / k_seed_dirty kernels above), so cost is proportional
    // to the front, not the whole mesh. This is the headline single-GPU optimization.
    void balance() {
        last_balance_iters = 0;

        int check_block, scatter_block, seed_block, d;
        cudaOccupancyMaxPotentialBlockSize(&d, &check_block, k_check_balance_active<DIM>, 0, 0);
        cudaOccupancyMaxPotentialBlockSize(&d, &scatter_block, k_scatter_refine, 0, 0);
        cudaOccupancyMaxPotentialBlockSize(&d, &seed_block, k_mark_child_neighbors<DIM>, 0, 0);

        thrust::device_vector<int> violation_flag(1);

        // dirty mask: pass 1 inspects every leaf. child_mask/dirty_next are reused.
        thrust::device_vector<uint8_t> dirty(codes.size(), 1);
        thrust::device_vector<uint8_t> child_mask;
        thrust::device_vector<uint8_t> dirty_next;
        bool front_collapsed = false;  // sticky: fall back to full checks if front grows large

        while (true) {
            int n = codes.size();

            aux_flags.resize(n);
            thrust::fill(aux_flags.begin(), aux_flags.end(), 0);
            violation_flag[0] = 0;

            int grid = (n + check_block - 1) / check_block;
            k_check_balance_active<DIM>
                <<<grid, check_block>>>(thrust::raw_pointer_cast(codes.data()),
                                        thrust::raw_pointer_cast(levels.data()),
                                        n,
                                        max_level,
                                        thrust::raw_pointer_cast(dirty.data()),
                                        thrust::raw_pointer_cast(aux_flags.data()),
                                        thrust::raw_pointer_cast(violation_flag.data()));
            CHECK_CUDA(cudaDeviceSynchronize());

            if (violation_flag[0] == 0)
                break;
            ++last_balance_iters;

            aux_counts.resize(n);
            aux_offsets.resize(n);
            thrust::transform(
                aux_flags.begin(), aux_flags.end(), aux_counts.begin(), [] __device__(int f) {
                    return f ? (1 << DIM) : 1;
                });
            thrust::exclusive_scan(aux_counts.begin(), aux_counts.end(), aux_offsets.begin());
            int total = aux_offsets.back() + aux_counts.back();

            scratch_codes.resize(total);
            scratch_levels.resize(total);

            int sgrid = (n + scatter_block - 1) / scatter_block;
            k_scatter_refine<<<sgrid, scatter_block>>>(
                thrust::raw_pointer_cast(codes.data()),
                thrust::raw_pointer_cast(levels.data()),
                n,
                thrust::raw_pointer_cast(aux_offsets.data()),
                thrust::raw_pointer_cast(aux_counts.data()),
                thrust::raw_pointer_cast(scratch_codes.data()),
                thrust::raw_pointer_cast(scratch_levels.data()),
                max_level,
                DIM);
            CHECK_CUDA(cudaDeviceSynchronize());

            // Seed the next pass's dirty mask over the NEW array. Front-tracking
            // only pays off when the refinement front stays a small fraction of the
            // mesh. For a large front (e.g. a crack plane), marking every child's
            // fine neighbours produces a dirty set nearly as big as the whole mesh,
            // so the seeding work is wasted -- the next check is no cheaper than a
            // full one. We detect this by measuring the seeded dirty count, and once
            // it is large we STICK to plain full checks for the rest of the balance
            // (front_collapsed). This keeps balance() >= balance_ref() on every
            // workload while winning big on thin features.
            // Cheap pre-filter: if this pass already refined a large fraction, the
            // front is wide -> skip seeding outright (children = refined*2^DIM is a
            // lower bound on the dirty set). The reduce below is the backstop for
            // fronts that are sparse in children yet wide after neighbour marking.
            long long refined = (long long)(total - n) / ((1 << DIM) - 1);
            if (front_collapsed || refined * (1 << DIM) * 64 > (long long)total) {
                dirty_next.assign(total, 1);  // plain full check, no seeding
            } else {
                dirty_next.assign(total, 0);
                // Pass A: mark new children (over OLD refined cells).
                child_mask.assign(total, 0);
                int childA_grid = (n + seed_block - 1) / seed_block;
                k_mark_children<<<childA_grid, seed_block>>>(
                    n,
                    thrust::raw_pointer_cast(aux_flags.data()),
                    thrust::raw_pointer_cast(aux_offsets.data()),
                    thrust::raw_pointer_cast(child_mask.data()),
                    DIM);
                CHECK_CUDA(cudaDeviceSynchronize());

                // Pass B: from each new child, mark its face-neighbours (over NEW cells).
                thrust::copy(child_mask.begin(), child_mask.end(), dirty_next.begin());
                int childB_grid = (total + seed_block - 1) / seed_block;
                k_mark_child_neighbors<DIM>
                    <<<childB_grid, seed_block>>>(thrust::raw_pointer_cast(scratch_codes.data()),
                                                  thrust::raw_pointer_cast(scratch_levels.data()),
                                                  total,
                                                  max_level,
                                                  thrust::raw_pointer_cast(child_mask.data()),
                                                  thrust::raw_pointer_cast(dirty_next.data()));
                CHECK_CUDA(cudaDeviceSynchronize());

                // If the resulting front is a large fraction of the mesh, stop
                // front-tracking for the remaining passes (cheap reduce; the
                // dirty set we just built is still correct to use this once).
                long long dirty_count = thrust::reduce(dirty_next.begin(), dirty_next.end(), 0LL);
                if (dirty_count * 8 > (long long)total)
                    front_collapsed = true;
            }

            codes.swap(scratch_codes);
            levels.swap(scratch_levels);
            dirty.swap(dirty_next);
        }
    }

    // Reference 2:1 balance: re-checks the whole mesh every pass. Simple and
    // obviously correct; kept as the parity oracle and benchmark baseline.
    void balance_ref() {
        last_balance_iters = 0;
        thrust::device_vector<int> violation_flag(1);

        int check_block, check_grid_dummy;
        int scatter_block, scatter_grid_dummy;
        cudaOccupancyMaxPotentialBlockSize(
            &check_grid_dummy, &check_block, k_check_balance<DIM>, 0, 0);
        cudaOccupancyMaxPotentialBlockSize(
            &scatter_grid_dummy, &scatter_block, k_scatter_refine, 0, 0);

        while (true) {
            int n = codes.size();

            // Reuse member vectors
            aux_flags.resize(n);
            thrust::fill(aux_flags.begin(), aux_flags.end(), 0);
            violation_flag[0] = 0;

            int grid = (n + check_block - 1) / check_block;

            k_check_balance<DIM>
                <<<grid, check_block>>>(thrust::raw_pointer_cast(codes.data()),
                                        thrust::raw_pointer_cast(levels.data()),
                                        n,
                                        max_level,
                                        thrust::raw_pointer_cast(aux_flags.data()),
                                        thrust::raw_pointer_cast(violation_flag.data()));
            CHECK_CUDA(cudaDeviceSynchronize());

            if (violation_flag[0] == 0)
                break;
            ++last_balance_iters;

            aux_counts.resize(n);
            aux_offsets.resize(n);

            thrust::transform(
                aux_flags.begin(), aux_flags.end(), aux_counts.begin(), [=] __device__(int f) {
                    return f ? (1 << DIM) : 1;
                });

            thrust::exclusive_scan(aux_counts.begin(), aux_counts.end(), aux_offsets.begin());
            int total = aux_offsets.back() + aux_counts.back();

            scratch_codes.resize(total);
            scratch_levels.resize(total);

            int scatter_grid = (n + scatter_block - 1) / scatter_block;

            k_scatter_refine<<<scatter_grid, scatter_block>>>(
                thrust::raw_pointer_cast(codes.data()),
                thrust::raw_pointer_cast(levels.data()),
                n,
                thrust::raw_pointer_cast(aux_offsets.data()),
                thrust::raw_pointer_cast(aux_counts.data()),
                thrust::raw_pointer_cast(scratch_codes.data()),
                thrust::raw_pointer_cast(scratch_levels.data()),
                max_level,
                DIM);
            CHECK_CUDA(cudaDeviceSynchronize());

            codes.swap(scratch_codes);
            levels.swap(scratch_levels);
        }
    }
};

using Quadtree = LinearTree<2>;
using Octree = LinearTree<3>;

}  // namespace amr