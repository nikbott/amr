#pragma once
#include "morton.hpp"
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <tuple>
#include <cassert>

struct Node {
    uint64_t code;
    int level;

    // Operator< is essential for std::lower_bound
    bool operator<(const Node& other) const { return code < other.code; }
    bool operator==(const Node& other) const { return code == other.code && level == other.level; }
};

template <int DIM>
class LinearTree {
public:
    int max_level;
    std::vector<Node> leaves;

    LinearTree(int max_lvl = 21) : max_level(max_lvl) {
        // Safety Check: 3D Morton codes packed into uint64_t cannot exceed level 21 
        // (3 bits * 21 levels = 63 bits).
        if constexpr (DIM == 3) {
            assert(max_level <= 21 && "3D Morton code overflows uint64_t above level 21");
        }
        leaves.push_back({0, 0});
    }

    uint64_t domain_width() const { return 1ULL << max_level; }

    // Helpers to bridge Morton2D/3D differences (Implemented below)
    std::vector<uint64_t> decode_coords(uint64_t code) const;
    uint64_t encode_coords(const std::vector<uint64_t>& coords) const;
    uint64_t get_neighbor_code(uint64_t code, int level, const std::vector<int>& offset) const;

    std::pair<std::vector<uint64_t>, uint64_t> get_geometry(const Node& node) const {
        auto coords = decode_coords(node.code);
        uint64_t size = 1ULL << (max_level - node.level);
        return {coords, size};
    }

    template <typename Oracle>
    bool refine(Oracle& oracle) {
        std::vector<Node> new_leaves;
        new_leaves.reserve(leaves.size()); 
        bool has_changed = false;

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
            leaves = std::move(new_leaves);
        }
        return has_changed;
    }

    template <typename Oracle>
    bool coarsen(Oracle& oracle) {
        std::vector<Node> new_leaves;
        new_leaves.reserve(leaves.size());
        bool has_changed = false;
        
        constexpr int num_siblings = 1 << DIM;
        size_t i = 0;
        
        while (i < leaves.size()) {
            bool should_coarsen = false;
            
            // Check if we have enough nodes for a family
            if (i + num_siblings <= leaves.size()) {
                const auto& first = leaves[i];
                int lvl = first.level;
                
                // Root cannot be coarsened
                if (lvl > 0) {
                    // Check 1: Are all nodes at the same level?
                    bool same_level = true;
                    for (int k = 1; k < num_siblings; ++k) {
                        if (leaves[i+k].level != lvl) {
                            same_level = false;
                            break;
                        }
                    }

                    // Check 2: Do they belong to the same parent?
                    // We check if the 0th child has coordinates divisible by parent size
                    // and if codes are consecutive.
                    bool aligned_start = false;
                    if (same_level) {
                        auto [coords, size] = get_geometry(first);
                        uint64_t parent_size = size * 2;
                        
                        bool coords_aligned = true;
                        for (auto c : coords) {
                            if (c % parent_size != 0) {
                                coords_aligned = false;
                                break;
                            }
                        }
                        aligned_start = coords_aligned;
                    }

                    if (same_level && aligned_start) {
                        // Construct potential parent
                        Node parent = {first.code, lvl - 1};
                        
                        // Oracle returns TRUE if it wants to REFINE.
                        // So for coarsening, we assume the oracle returns FALSE (no refinement needed).
                        if (!oracle(parent, max_level)) {
                            new_leaves.push_back(parent);
                            i += num_siblings;
                            has_changed = true;
                            should_coarsen = true;
                        }
                    }
                }
            }
            
            if (!should_coarsen) {
                new_leaves.push_back(leaves[i]);
                i++;
            }
        }

        if (has_changed) {
            // No need to sort if we process in order and append
            leaves = std::move(new_leaves);
            // Release unused memory after coarsening
            leaves.shrink_to_fit();
        }
        return has_changed;
    }

    void balance() {
        // 2:1 Balance Ripple Algorithm using Binary Search
        // Iterates until equilibrium (no more violations). 
        // Guaranteed to terminate as level cannot exceed max_level.
        
        std::vector<std::vector<int>> directions;
        if constexpr (DIM == 2) {
            directions = {{1,0}, {-1,0}, {0,1}, {0,-1}};
        } else {
            directions = {{1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1}};
        }

        while (true) {
            // In a Linear Tree, 'leaves' is always sorted by Morton code.
            // We use this property to perform binary search.
            
            std::unordered_set<uint64_t> to_refine_codes;

            for (const auto& node : leaves) {
                // If I am at level L, I trigger refinement in neighbors 
                // who are at level <= L - 2.
                
                for (const auto& dir : directions) {
                    // Get theoretical neighbor code at SAME level
                    uint64_t base_n_code = get_neighbor_code(node.code, node.level, dir);
                    if (base_n_code == UINT64_MAX) continue; // Boundary

                    // Search for this neighbor in the tree.
                    // It might be coarser. We check levels node.level-2 down to 0.
                    int search_lvl = node.level - 2;
                    while (search_lvl >= 0) {
                        // Mask the base_n_code to get the ancestor code at search_lvl.
                        int shift_bits = (max_level - search_lvl) * DIM;
                        uint64_t mask = (shift_bits >= 64) ? 0 : (~0ULL << shift_bits);
                        uint64_t coarse_n_code = base_n_code & mask;

                        if (to_refine_codes.count(coarse_n_code)) break; // Already handled

                        // Binary Search for neighbor
                        Node target = {coarse_n_code, 0}; // Level is dummy for comparison
                        auto it = std::lower_bound(leaves.begin(), leaves.end(), target);

                        if (it != leaves.end() && it->code == coarse_n_code) {
                            // We found a leaf with this Morton code.
                            // Check if it is the coarser neighbor we are worried about
                            if (it->level == search_lvl) {
                                // Violation detected: Neighbor is coarser by >= 2 levels.
                                to_refine_codes.insert(it->code);
                                break; 
                            }
                            // If code matches but level != search_lvl, the neighbor is FINER.
                            // This is not a violation for the current node.
                        }
                        search_lvl--;
                    }
                }
            }

            if (to_refine_codes.empty()) break; // Equilibrium reached

            auto oracle = [&](const Node& n, int) {
                return to_refine_codes.count(n.code) > 0;
            };
            refine(oracle);
        }
    }
};

// Template Specializations
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

using Quadtree = LinearTree<2>;
using Octree   = LinearTree<3>;