#ifdef USE_CUDA

#include "tree_kernels.cuh"
#include "cuda_utils.cuh"
#include "physics.hpp" // For CircleOracleData
#include <climits>

// ============================================================================
// MORTON GPU IMPLEMENTATION
// ============================================================================

namespace Morton2D_GPU {
    __device__ __forceinline__ uint64_t spread(uint64_t n) {
        const uint64_t MASK_1 = 0x00000000FFFFFFFF;
        const uint64_t MASK_2 = 0x0000FFFF0000FFFF;
        const uint64_t MASK_3 = 0x00FF00FF00FF00FF;
        const uint64_t MASK_4 = 0x0F0F0F0F0F0F0F0F;
        const uint64_t MASK_5 = 0x3333333333333333;
        const uint64_t MASK_6 = 0x5555555555555555;
        
        n &= MASK_1;
        n = (n | (n << 16)) & MASK_2;
        n = (n | (n << 8))  & MASK_3;
        n = (n | (n << 4))  & MASK_4;
        n = (n | (n << 2))  & MASK_5;
        n = (n | (n << 1))  & MASK_6;
        return n;
    }

    __device__ __forceinline__ uint64_t compact(uint64_t n) {
        const uint64_t MASK_1 = 0x00000000FFFFFFFF;
        const uint64_t MASK_2 = 0x0000FFFF0000FFFF;
        const uint64_t MASK_3 = 0x00FF00FF00FF00FF;
        const uint64_t MASK_4 = 0x0F0F0F0F0F0F0F0F;
        const uint64_t MASK_5 = 0x3333333333333333;
        const uint64_t MASK_6 = 0x5555555555555555;
        
        n &= MASK_6;
        n = (n | (n >> 1)) & MASK_5;
        n = (n | (n >> 2)) & MASK_4;
        n = (n | (n >> 4)) & MASK_3;
        n = (n | (n >> 8)) & MASK_2;
        n = (n | (n >> 16)) & MASK_1;
        return n;
    }

    __device__ __forceinline__ uint64_t encode(uint64_t x, uint64_t y) {
        return (spread(y) << 1) | spread(x);
    }

    __device__ __forceinline__ void decode(uint64_t code, uint64_t& x, uint64_t& y) {
        x = compact(code);
        y = compact(code >> 1);
    }

    __device__ __forceinline__ uint64_t get_neighbor(uint64_t code, int level, int dx, int dy, int max_level) {
        uint64_t x, y;
        decode(code, x, y);
        
        // Size of the cell at this level
        uint64_t size = 1ULL << (max_level - level);

        
        long long nx = (long long)x + (long long)dx * size;
        long long ny = (long long)y + (long long)dy * size;
        
        long long limit = 1ULL << max_level;
        
        if (nx < 0 || nx >= limit || ny < 0 || ny >= limit) {
            return UINT64_MAX;
        }
        
        return encode((uint64_t)nx, (uint64_t)ny);
    }
}

// Binary Search to find the node that contains the given code
// Returns index of the node, or -1 if not found
__device__ int findNodeIndex(const uint64_t* codes, const int* levels, int n, uint64_t target_code, int max_level) {
    int left = 0;
    int right = n - 1;
    int ans = -1;
    
    // Find the last node with code <= target_code (upper_bound - 1)
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (codes[mid] <= target_code) {
            ans = mid;
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    if (ans != -1) {

        int level = levels[ans];
        int shift = 2 * (max_level - level);
        // Be careful with 64-bit shift
        uint64_t range = (shift >= 64) ? UINT64_MAX : (1ULL << shift);
        
        if (target_code < codes[ans] + range) {
            return ans;
        }
    }
    
    return -1;
}

// ============================================================================
// PARALLEL PRIMITIVES (PURE CUDA)
// ============================================================================

// Parallel Bitonic Sort Step
__global__ void bitonicSortStep(uint64_t* codes, int* levels, int n, int j, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ixj = idx ^ j;
    
    if (ixj > idx && idx < n && ixj < n) {
        if ((idx & k) == 0) {
            // Ascending
            if (codes[idx] > codes[ixj]) {
                // Swap
                uint64_t tc = codes[idx]; codes[idx] = codes[ixj]; codes[ixj] = tc;
                int tl = levels[idx]; levels[idx] = levels[ixj]; levels[ixj] = tl;
            }
        } else {
            // Descending
            if (codes[idx] < codes[ixj]) {
                // Swap
                uint64_t tc = codes[idx]; codes[idx] = codes[ixj]; codes[ixj] = tc;
                int tl = levels[idx]; levels[idx] = levels[ixj]; levels[ixj] = tl;
            }
        }
    }
}

void bitonicSortGPU(uint64_t* d_codes, int* d_levels, int n) {
    // Round up to power of 2
    int n_padded = 1;
    while (n_padded < n) n_padded *= 2;
    
    // Pad if needed
    if (n_padded > n) {
        uint64_t* h_max_codes = new uint64_t[n_padded - n];
        int* h_max_levels = new int[n_padded - n];
        for (int i = 0; i < n_padded - n; i++) {
            h_max_codes[i] = UINT64_MAX;
            h_max_levels[i] = INT_MAX;
        }
        CUDA_CHECK(cudaMemcpy(d_codes + n, h_max_codes, (n_padded - n) * sizeof(uint64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_levels + n, h_max_levels, (n_padded - n) * sizeof(int), cudaMemcpyHostToDevice));
        delete[] h_max_codes;
        delete[] h_max_levels;
    }
    
    int blockSize = cuda_utils::getOptimalBlockSize();
    int gridSize = cuda_utils::getGridSize(n_padded, blockSize);
    
    for (int k = 2; k <= n_padded; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            bitonicSortStep<<<gridSize, blockSize>>>(d_codes, d_levels, n_padded, j, k);
            CUDA_CHECK_LAST();
        }
    }
}

// Parallel Exclusive Scan (Blelloch)
__global__ void scanUpSweep(int* data, int n, int stride) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride * 2;
    if (idx + stride < n) {
        data[idx + stride * 2 - 1] += data[idx + stride - 1];
    }
}

__global__ void scanDownSweep(int* data, int n, int stride) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride * 2;
    if (idx + stride < n) {
        int temp = data[idx + stride - 1];
        data[idx + stride - 1] = data[idx + stride * 2 - 1];
        data[idx + stride * 2 - 1] += temp;
    }
}
// Scan orchestrator creates sweep kernels
void exclusiveScanGPU(int* d_data, int n) {
    int n_padded = 1;
    while (n_padded < n) n_padded *= 2;
    
    if (n_padded > n) {
        CUDA_CHECK(cudaMemset(d_data + n, 0, (n_padded - n) * sizeof(int)));
    }
    
    int blockSize = cuda_utils::getOptimalBlockSize();
    
    // Up-sweep
    for (int stride = 1; stride < n_padded; stride *= 2) {
        int gridSize = cuda_utils::getGridSize(n_padded / (stride * 2), blockSize);
        if (gridSize > 0) {
            scanUpSweep<<<gridSize, blockSize>>>(d_data, n_padded, stride);
            CUDA_CHECK_LAST();
        }
    }
    
    // Set last to 0
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(d_data + n_padded - 1, &zero, sizeof(int), cudaMemcpyHostToDevice));
    
    // Down-sweep
    for (int stride = n_padded / 2; stride >= 1; stride /= 2) {
        int gridSize = cuda_utils::getGridSize(n_padded / (stride * 2), blockSize);
        if (gridSize > 0) {
            scanDownSweep<<<gridSize, blockSize>>>(d_data, n_padded, stride);
            CUDA_CHECK_LAST();
        }
    }
}

// ============================================================================
// ORACLE & REFINE KERNELS
// ============================================================================

__global__ void evaluateOracleKernel(
    CircleOracleData oracle,
    const uint64_t* codes,
    const int* levels,
    bool* results,
    int n,
    int max_level
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        uint64_t code = codes[idx];
        int level = levels[idx];
        
        uint64_t x, y;
        Morton2D_GPU::decode(code, x, y);
        
        uint64_t size = 1ULL << (max_level - level);
        
        double node_cx = x + size * 0.5;
        double node_cy = y + size * 0.5;
        
        double dx = node_cx - oracle.cx;
        double dy = node_cy - oracle.cy;
        double dist_sq = dx*dx + dy*dy;
        
        double extent = size * 0.70710678;
        double threshold = oracle.bandwidth + extent;
        
        double r = (double)oracle.radius;
        double upper = r + threshold;
        double lower = r - threshold;
        
        double upper_sq = upper * upper;
        double lower_sq = (lower < 0) ? 0.0 : lower * lower;
        
        bool is_refining = (dist_sq < upper_sq) && (dist_sq > lower_sq);
        
        results[idx] = (level < oracle.coarse_level) || 
                       (is_refining && level < oracle.fine_level);
    }
}

__global__ void markRefinementKernel(
    const bool* oracle_results,
    int* refine_flags,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        refine_flags[idx] = oracle_results[idx] ? 1 : 0;
    }
}

__global__ void countOutputsKernel(
    const int* refine_flags,
    int* output_counts,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output_counts[idx] = refine_flags[idx] ? 4 : 1;
    }
}

__global__ void expandNodesKernel(
    const uint64_t* parent_codes,
    const int* parent_levels,
    const int* refine_flags,
    const int* scan_results,
    uint64_t* child_codes,
    int* child_levels,
    int n,
    int max_level
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (refine_flags[idx]) {
            uint64_t x, y;
            Morton2D_GPU::decode(parent_codes[idx], x, y);
            
            int new_level = parent_levels[idx] + 1;
            uint64_t step = 1ULL << (max_level - new_level);
            
            int out_base = scan_results[idx]; // Scan gives exclusive sum, so this is start index
            // But wait, scan is on output_counts (4 or 1).
            // If we are thread i, scan[i] is the start index for our children.
            
            // Child 0
            child_codes[out_base + 0] = Morton2D_GPU::encode(x, y);
            child_levels[out_base + 0] = new_level;
            
            // Child 1
            child_codes[out_base + 1] = Morton2D_GPU::encode(x + step, y);
            child_levels[out_base + 1] = new_level;
            
            // Child 2
            child_codes[out_base + 2] = Morton2D_GPU::encode(x, y + step);
            child_levels[out_base + 2] = new_level;
            
            // Child 3
            child_codes[out_base + 3] = Morton2D_GPU::encode(x + step, y + step);
            child_levels[out_base + 3] = new_level;
        } else {
            int out_idx = scan_results[idx];
            child_codes[out_idx] = parent_codes[idx];
            child_levels[out_idx] = parent_levels[idx];
        }
    }
}

void refineCUDA(
    uint64_t** d_codes_ptr,
    int** d_levels_ptr,
    int* n_nodes,
    const void* oracle_data_ptr,
    int max_level
) {
    int n = *n_nodes;
    if (n == 0) return;
    
    // Cast de tipo do Oracle, permite a adição de outros oracles no futuro
    const CircleOracleData* oracle = (const CircleOracleData*)oracle_data_ptr;
    
    uint64_t* d_codes = *d_codes_ptr;
    int* d_levels = *d_levels_ptr;
    
    // Allocate temps
    bool* d_oracle_results;
    int* d_refine_flags;
    int* d_output_counts;
    int* d_scan;
    
    CUDA_CHECK(cudaMalloc(&d_oracle_results, n * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_refine_flags, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output_counts, n * sizeof(int)));

    // Allocate scan array with padding for power of 2
    int n_padded = 1;
    while (n_padded < n) n_padded *= 2;
    CUDA_CHECK(cudaMalloc(&d_scan, n_padded * sizeof(int)));
    
    int blockSize = cuda_utils::getOptimalBlockSize();
    int gridSize = cuda_utils::getGridSize(n, blockSize);
    
    // 1. Evaluate Oracle
    evaluateOracleKernel<<<gridSize, blockSize>>>(
        *oracle, d_codes, d_levels, d_oracle_results, n, max_level
    );
    CUDA_CHECK_LAST();
    
    // 2. Mark Refinement
    markRefinementKernel<<<gridSize, blockSize>>>(
        d_oracle_results, d_refine_flags, n
    );
    CUDA_CHECK_LAST();
    
    // 3. Count Outputs
    countOutputsKernel<<<gridSize, blockSize>>>(
        d_refine_flags, d_output_counts, n
    );
    CUDA_CHECK_LAST();
    
    // 4. Scan
    // Copy counts to scan array
    CUDA_CHECK(cudaMemcpy(d_scan, d_output_counts, n * sizeof(int), cudaMemcpyDeviceToDevice));
    exclusiveScanGPU(d_scan, n);
    
    // 5. Calculate Total Size
    int last_scan, last_count;
    CUDA_CHECK(cudaMemcpy(&last_scan, d_scan + n - 1, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&last_count, d_output_counts + n - 1, sizeof(int), cudaMemcpyDeviceToHost));
    int total_outputs = last_scan + last_count;
    
    if (total_outputs == n) {
        // No changes
        cudaFree(d_oracle_results);
        cudaFree(d_refine_flags);
        cudaFree(d_output_counts);
        cudaFree(d_scan);
        return;
    }
    
    // 6. Allocate New Arrays
    int total_outputs_padded = 1;
    while (total_outputs_padded < total_outputs) total_outputs_padded *= 2;

    uint64_t* d_new_codes;
    int* d_new_levels;
    CUDA_CHECK(cudaMalloc(&d_new_codes, total_outputs_padded * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_new_levels, total_outputs_padded * sizeof(int)));
    
    // 7. Expand
    expandNodesKernel<<<gridSize, blockSize>>>(
        d_codes, d_levels, d_refine_flags, d_scan,
        d_new_codes, d_new_levels, n, max_level
    );
    CUDA_CHECK_LAST();
    
    // 8. Sort
    bitonicSortGPU(d_new_codes, d_new_levels, total_outputs);
    
    // Swap pointers
    cudaFree(d_codes);
    cudaFree(d_levels);
    *d_codes_ptr = d_new_codes;
    *d_levels_ptr = d_new_levels;
    *n_nodes = total_outputs;
    
    // Cleanup
    cudaFree(d_oracle_results);
    cudaFree(d_refine_flags);
    cudaFree(d_output_counts);
    cudaFree(d_scan);
}

__global__ void checkBalanceKernel(
    const uint64_t* codes,
    const int* levels,
    int* refine_flags,
    int n,
    int max_level
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int level = levels[idx];
        
        // If level is too small, we can't have a neighbor at level-2
        if (level < 2) return;
        
        uint64_t code = codes[idx];
        
        // Check 8 neighbors
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                if (dx == 0 && dy == 0) continue;
                
                uint64_t n_code = Morton2D_GPU::get_neighbor(code, level, dx, dy, max_level);
                
                if (n_code != UINT64_MAX) {
                    // Find who covers this neighbor code
                    int n_idx = findNodeIndex(codes, levels, n, n_code, max_level);
                    
                    if (n_idx != -1) {
                        int n_level = levels[n_idx];

                        // Constraint: Neighbor cannot be more than 1 level coarser
                        // If n_level <= level - 2, then neighbor is too coarse.
                        // Neighbor must refine.

                        if (n_level <= level - 2) {
                            // Mark neighbor for refinement
                            atomicExch(&refine_flags[n_idx], 1);
                        }
                    }
                }
            }
        }
    }
}

void balanceCUDA(
    uint64_t** d_codes_ptr,
    int** d_levels_ptr,
    int* n_nodes,
    int max_level,
    int max_iter
) {
    for (int iter = 0; iter < max_iter; ++iter) {
        int n = *n_nodes;
        if (n == 0) break;
        
        uint64_t* d_codes = *d_codes_ptr;
        int* d_levels = *d_levels_ptr;
        
        // Allocate temps
        int* d_refine_flags;
        int* d_output_counts;
        int* d_scan;
        
        CUDA_CHECK(cudaMalloc(&d_refine_flags, n * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_refine_flags, 0, n * sizeof(int))); // Init to 0
        CUDA_CHECK(cudaMalloc(&d_output_counts, n * sizeof(int)));
        
        int n_padded = 1;
        while (n_padded < n) n_padded *= 2;
        CUDA_CHECK(cudaMalloc(&d_scan, n_padded * sizeof(int)));
        
        int blockSize = cuda_utils::getOptimalBlockSize();
        int gridSize = cuda_utils::getGridSize(n, blockSize);
        
        // 1. Check Balance
        checkBalanceKernel<<<gridSize, blockSize>>>(
            d_codes, d_levels, d_refine_flags, n, max_level
        );
        CUDA_CHECK_LAST();
        
        // 2. Count Outputs
        countOutputsKernel<<<gridSize, blockSize>>>(
            d_refine_flags, d_output_counts, n
        );
        CUDA_CHECK_LAST();
        
        // 3. Scan
        CUDA_CHECK(cudaMemcpy(d_scan, d_output_counts, n * sizeof(int), cudaMemcpyDeviceToDevice));
        exclusiveScanGPU(d_scan, n);
        
        // 4. Calculate Total Size
        int last_scan, last_count;
        CUDA_CHECK(cudaMemcpy(&last_scan, d_scan + n - 1, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&last_count, d_output_counts + n - 1, sizeof(int), cudaMemcpyDeviceToHost));
        int total_outputs = last_scan + last_count;
        
        if (total_outputs == n) {
            // No changes
            cudaFree(d_refine_flags);
            cudaFree(d_output_counts);
            cudaFree(d_scan);
            break;
        }
        
        // 5. Allocate New Arrays
        int total_outputs_padded = 1;
        while (total_outputs_padded < total_outputs) total_outputs_padded *= 2;

        uint64_t* d_new_codes;
        int* d_new_levels;
        CUDA_CHECK(cudaMalloc(&d_new_codes, total_outputs_padded * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_new_levels, total_outputs_padded * sizeof(int)));
        
        // 6. Expand
        expandNodesKernel<<<gridSize, blockSize>>>(
            d_codes, d_levels, d_refine_flags, d_scan,
            d_new_codes, d_new_levels, n, max_level
        );
        CUDA_CHECK_LAST();
        
        // 7. Sort
        bitonicSortGPU(d_new_codes, d_new_levels, total_outputs);
        
        // Swap pointers
        cudaFree(d_codes);
        cudaFree(d_levels);
        *d_codes_ptr = d_new_codes;
        *d_levels_ptr = d_new_levels;
        *n_nodes = total_outputs;
        
        // Cleanup
        cudaFree(d_refine_flags);
        cudaFree(d_output_counts);
        cudaFree(d_scan);
    }
}

#endif // USE_CUDA
