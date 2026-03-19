#pragma once
#include <vector>
#include <cstdint>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <type_traits>
#include "morton.hpp"

class CircleOracle2D;

#ifdef USE_CUDA

#include <cuda_runtime.h>
void refineCUDA(uint64_t** d_codes_ptr, int** d_levels_ptr, int* n_nodes,
                     const void* oracle_data, int max_level);
void balanceCUDA(uint64_t** d_codes_ptr, int** d_levels_ptr, int* n_nodes,
                      int max_level, int max_iter);
#endif

struct Node {
    uint64_t code;
    int level;

    bool operator<(const Node& other) const {
        return code < other.code;
    }
    bool operator==(const Node& other) const {
        return code == other.code && level == other.level;
    }
};


template <int DIM>
class LinearTree {
public:
    int max_level;
    std::vector<Node> leaves;
    bool use_gpu;

#ifdef USE_CUDA
    uint64_t* d_codes = nullptr;
    int* d_levels = nullptr;
    int d_size = 0;
    bool gpu_dirty = false;
#endif

    LinearTree(int max_lvl = 21, bool use_gpu_flag = false) 
        : max_level(max_lvl), use_gpu(use_gpu_flag) {
        leaves.push_back({0, 0});
#ifdef USE_CUDA
        if (use_gpu) {
            syncToGPU();
        }
#endif
    }

    ~LinearTree() {
#ifdef USE_CUDA
        if (d_codes) cudaFree(d_codes);
        if (d_levels) cudaFree(d_levels);
#endif
    }

    uint64_t domain_width() const { return 1ULL << max_level; }

    // Helpers to bridge Morton2D/3D differences
    std::vector<uint64_t> decode_coords(uint64_t code) const;
    uint64_t encode_coords(const std::vector<uint64_t>& coords) const;
    uint64_t get_neighbor_code(uint64_t code, int level, const std::vector<int>& offset) const;

    // Returns (coords, size)
    std::pair<std::vector<uint64_t>, uint64_t> get_geometry(const Node& node) const {
        auto coords = decode_coords(node.code);
        uint64_t size = 1ULL << (max_level - node.level);
        return {coords, size};
    }

#ifdef USE_CUDA
    void syncToGPU() {
        if (d_codes) cudaFree(d_codes);
        if (d_levels) cudaFree(d_levels);
        
        d_size = leaves.size();
        cudaMalloc(&d_codes, d_size * sizeof(uint64_t));
        cudaMalloc(&d_levels, d_size * sizeof(int));
        
        std::vector<uint64_t> codes(d_size);
        std::vector<int> levels(d_size);
        for(int i=0; i<d_size; ++i) {
            codes[i] = leaves[i].code;
            levels[i] = leaves[i].level;
        }
        
        cudaMemcpy(d_codes, codes.data(), d_size * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_levels, levels.data(), d_size * sizeof(int), cudaMemcpyHostToDevice);
        gpu_dirty = false;
    }

    void syncFromGPU() {
        leaves.resize(d_size);
        std::vector<uint64_t> codes(d_size);
        std::vector<int> levels(d_size);
        
        cudaMemcpy(codes.data(), d_codes, d_size * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(levels.data(), d_levels, d_size * sizeof(int), cudaMemcpyDeviceToHost);
        
        for(int i=0; i<d_size; ++i) {
            leaves[i] = {codes[i], levels[i]};
        }
        gpu_dirty = false;
    }

    int getGPUSize() const {
        return d_size;
    }
#endif


    template <typename Oracle>
    bool refine(Oracle& oracle) {
#ifdef USE_CUDA
       if (use_gpu) {
            if constexpr (std::is_same_v<Oracle, CircleOracle2D>) {
                if (gpu_dirty) syncToGPU();
                
                auto oracle_data = oracle.get_gpu_data(max_level);
                int before = d_size;
                refineCUDA(&d_codes, &d_levels, &d_size, &oracle_data, max_level);
                
                return d_size != before;
            }
        }
#endif

        std::vector<Node> new_leaves;
        // Optimization: Reserve memory
        new_leaves.reserve(leaves.size()); 
        
        bool has_changed = false;

        // Base offsets for children
        std::vector<std::vector<int>> offsets;
        if constexpr (DIM == 2) {
            offsets = {{0,0}, {1,0}, {0,1}, {1,1}};
        } else {
            offsets = {{0,0,0}, {1,0,0}, {0,1,0}, {1,1,0},
                       {0,0,1}, {1,0,1}, {0,1,1}, {1,1,1}};
        }

        for (const auto& node : leaves) {
            if (oracle(node, max_level)) {
                has_changed = true;
                auto current_coords = decode_coords(node.code);
                int new_lvl = node.level + 1;
                uint64_t step = 1ULL << (max_level - new_lvl);

                for (const auto& offset : offsets) {
                    std::vector<uint64_t> child_coords = current_coords;
                    for(size_t i=0; i<DIM; ++i) child_coords[i] += offset[i] * step;
                    
                    new_leaves.push_back({encode_coords(child_coords), new_lvl});
                }
            } else {
                new_leaves.push_back(node);
            }
        }

        if (has_changed) {
            std::sort(new_leaves.begin(), new_leaves.end());
            leaves = std::move(new_leaves);
        }
        return has_changed;
    }

    // Balance (2:1 Constraint)
    void balance() {
#ifdef USE_CUDA
        if (use_gpu) {
            if (gpu_dirty) syncToGPU();
            balanceCUDA(&d_codes, &d_levels, &d_size, max_level, 20);
            return;
        }
#endif
        int max_iter = 20;

        // Generate direction vectors (excluding 0,0...)
        std::vector<std::vector<int>> directions;
        if constexpr (DIM == 2) {
            for(int x=-1; x<=1; ++x) for(int y=-1; y<=1; ++y)
                if(x!=0 || y!=0) directions.push_back({x, y});
        } else {
            for(int x=-1; x<=1; ++x) for(int y=-1; y<=1; ++y) for(int z=-1; z<=1; ++z)
                if(x!=0 || y!=0 || z!=0) directions.push_back({x, y, z});
        }

        for (int iter = 0; iter < max_iter; ++iter) {
           
            std::unordered_map<uint64_t, Node> node_map;
            node_map.reserve(leaves.size());
            for(const auto& n : leaves) node_map[n.code] = n;

            std::unordered_set<uint64_t> to_refine_codes;

            // 2. Scan
            for (const auto& node : leaves) {
                int min_valid_level = node.level - 1;
                if (min_valid_level < 1) continue;

                for (const auto& dir : directions) {
                    uint64_t base_n_code = get_neighbor_code(node.code, node.level, dir);
                    if (base_n_code == UINT64_MAX) continue;

                    // Search coarser levels
                    int curr_search_level = node.level - 2;
                    while (curr_search_level >= 0) {
                        int shift_bits = (max_level - curr_search_level) * DIM;
                        uint64_t mask = ~((1ULL << shift_bits) - 1ULL);

                        uint64_t coarse_n_code = base_n_code & mask;

                        if (to_refine_codes.count(coarse_n_code)) break;

                        auto it = node_map.find(coarse_n_code);
                        if (it != node_map.end()) {
                            if (it->second.level == curr_search_level) {
                                to_refine_codes.insert(it->second.code);
                                break;
                            }
                        }
                        curr_search_level--;
                    }
                }
            }

            // 3. Batch Refine
            if (to_refine_codes.empty()) break;

            // Lambda for refinement
            auto refine_predicate = [&](const Node& n, int) {
                return to_refine_codes.count(n.code) > 0;
            };
            refine(refine_predicate);
        }
    }
};

// Template specializations for 2D/3D specifics
template<> inline std::vector<uint64_t> LinearTree<2>::decode_coords(uint64_t code) const {
    auto [x, y] = Morton2D::decode(code);
    return {x, y};
}
template<> inline uint64_t LinearTree<2>::encode_coords(const std::vector<uint64_t>& c) const {
    return Morton2D::encode((uint32_t)c[0], (uint32_t)c[1]);
}
template<> inline uint64_t LinearTree<2>::get_neighbor_code(uint64_t code, int level, const std::vector<int>& off) const {
    return Morton2D::get_neighbor(code, level, off[0], off[1], max_level);
}

template<> inline std::vector<uint64_t> LinearTree<3>::decode_coords(uint64_t code) const {
    auto [x, y, z] = Morton3D::decode(code);
    return {x, y, z};
}
template<> inline uint64_t LinearTree<3>::encode_coords(const std::vector<uint64_t>& c) const {
    return Morton3D::encode((uint32_t)c[0], (uint32_t)c[1], (uint32_t)c[2]);
}
template<> inline uint64_t LinearTree<3>::get_neighbor_code(uint64_t code, int level, const std::vector<int>& off) const {
    return Morton3D::get_neighbor(code, level, off[0], off[1], off[2], max_level);
}

// Aliases
using Quadtree = LinearTree<2>;
using Octree   = LinearTree<3>;
