#pragma once
#include <vector>
#include <cstdint>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <type_traits>
#include <omp.h>
#include "morton.hpp"

class CircleOracle2D;

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

// Abstract Linear Tree logic implemented via Templates
template <int DIM>
class LinearTree {
public:
    int max_level;
    std::vector<Node> leaves;

    LinearTree(int max_lvl = 21) 
        : max_level(max_lvl) {
        leaves.push_back({0, 0});
    }

    ~LinearTree() {}

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

    // Refine
    template <typename Oracle>
    bool refine(Oracle& oracle) {
        size_t n = leaves.size();
        std::vector<uint8_t> refine_flags(n, 0);
        bool any_refine = false;

        // 1. Compute refinement criteria in parallel
        #pragma omp parallel for schedule(dynamic, 1024) reduction(|:any_refine)
        for (size_t i = 0; i < n; i++) {
            if (oracle(leaves[i], max_level)) {
                refine_flags[i] = 1;
                any_refine = true;
            }
        }

        if (!any_refine) return false;

        // 2. Collect new cells using thread-local buffers
        std::vector<std::vector<Node>> thread_cells(omp_get_max_threads());

        // Base offsets for children
        std::vector<std::vector<int>> offsets;
        if constexpr (DIM == 2) {
            offsets = {{0,0}, {1,0}, {0,1}, {1,1}};
        } else {
            offsets = {{0,0,0}, {1,0,0}, {0,1,0}, {1,1,0},
                       {0,0,1}, {1,0,1}, {0,1,1}, {1,1,1}};
        }

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            // Heuristic reservation
            thread_cells[tid].reserve(n / omp_get_num_threads() * 2);

            #pragma omp for schedule(static) nowait
            for (size_t i = 0; i < n; i++) {
                const auto& node = leaves[i];
                if (refine_flags[i]) {
                    auto current_coords = decode_coords(node.code);
                    int new_lvl = node.level + 1;
                    uint64_t step = 1ULL << (max_level - new_lvl);

                    for (const auto& offset : offsets) {
                        std::vector<uint64_t> child_coords = current_coords;
                        for(size_t d=0; d<DIM; ++d) child_coords[d] += offset[d] * step;
                        
                        thread_cells[tid].push_back({encode_coords(child_coords), new_lvl});
                    }
                } else {
                    thread_cells[tid].push_back(node);
                }
            }
        }

        // 3. Serial merge (efficient enough for vectors)
        std::vector<Node> new_leaves;
        size_t total_size = 0;
        for (const auto& tc : thread_cells) total_size += tc.size();
        new_leaves.reserve(total_size);

        for (const auto& tc : thread_cells) {
            new_leaves.insert(new_leaves.end(), tc.begin(), tc.end());
        }

        // 4. Sort
        // Using std::sort (serial) for now, could be __gnu_parallel::sort if available
        std::sort(new_leaves.begin(), new_leaves.end());
        
        leaves = std::move(new_leaves);
        return true;
    }

    // Balance (2:1 Constraint)
    void balance() {
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
            // 1. Parallel Neighbor Search
            // Since leaves are sorted, we can use Binary Search instead of building a Map
            std::vector<std::vector<uint64_t>> thread_refine_codes(omp_get_max_threads());

            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                
                #pragma omp for schedule(dynamic, 512) nowait
                for (size_t i = 0; i < leaves.size(); i++) {
                    const auto& node = leaves[i];
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

                            // Binary Search for the neighbor using lower_bound
                            Node target_node = {coarse_n_code, 0};
                            auto it = std::lower_bound(leaves.begin(), leaves.end(), target_node);
                            
                            if (it != leaves.end() && it->code == coarse_n_code) {
                                if (it->level == curr_search_level) {
                                    thread_refine_codes[tid].push_back(it->code);
                                    break;
                                }
                            }
                            curr_search_level--;
                        }
                    }
                }
            }

            // 2. Deduplicate candidates
            std::unordered_set<uint64_t> to_refine_codes;
            for (const auto& tc : thread_refine_codes) {
                for (auto code : tc) to_refine_codes.insert(code);
            }

            if (to_refine_codes.empty()) break;

            // 3. Batch Refine
            // We can optimize this predicate lookup too if needed, but unordered_set is O(1)
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

