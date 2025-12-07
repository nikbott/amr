#pragma once
#include "core.hpp"
#include <vector>
#include <array>
#include <cassert>
#include <omp.h>

namespace amr {

/**
 * @brief Represents a leaf node in the Linear Octree.
 * * A node is defined by its Morton code (which encodes the anchor coordinate)
 * and its refinement level. In the Linear Octree method (Burstedde et al., 2011),
 * only leaf nodes are stored, sorted by their Morton code (Space-Filling Curve index).
 */
struct Node {
    uint64_t code;
    int level;

    // Sorting by code is the fundamental invariant of the Linear Octree storage scheme.
    bool operator<(const Node& other) const { return code < other.code; }
    bool operator==(const Node& other) const { return code == other.code && level == other.level; }
};

/**
 * @brief C++20 Concept for Refinement/Coarsening Criteria.
 * * Corresponds to the "callback function" described in p4est (Holke, 2018).
 * The oracle returns true if a node should be refined (or cannot be coarsened).
 */
template<typename T>
concept RefinementOracle = requires(T t, const Node& n, int max_lvl) {
    { t(n, max_lvl) } -> std::convertible_to<bool>;
};

/**
 * @brief Linear Tree (Quadtree/Octree) container.
 * * Implements the "Linear Octree" storage scheme where the mesh is represented
 * as a flat, sorted array of leaf nodes. This structure supports efficient
 * parallel traversals and "lock-free" mesh adaptation.
 * * @tparam DIM Dimension of the tree (2 for Quadtree, 3 for Octree).
 */
template <int DIM>
class LinearTree {
    static_assert(DIM == 2 || DIM == 3, "Only 2D or 3D trees supported.");

public:
    using Point = std::array<uint64_t, DIM>;
    
    /**
     * @brief Maximum depth of the tree.
     * Limited by 64-bit integer size for Morton codes.
     * 2D: max 31 levels. 3D: max 21 levels.
     */
    const int max_level;
    
    /// The linear array of leaf nodes, sorted by Morton code (SFC index).
    std::vector<Node> leaves;

    explicit LinearTree(int max_lvl) : max_level(max_lvl) {
        assert(max_lvl > 0);
        if constexpr (DIM == 3) assert(max_lvl <= 21);
        else assert(max_lvl <= 31);
        
        // Initialize with a single root node covering the entire domain.
        leaves.push_back({0, 0});
    }

    [[nodiscard]] uint64_t domain_width() const { return 1ULL << max_level; }

    // --- Geometry Helpers ---

    /**
     * @brief Decodes a Morton code into integer coordinates.
     */
    [[nodiscard]] Point decode(uint64_t code) const {
        if constexpr (DIM == 2) {
            auto [x, y] = morton::decode_2d(code);
            return {x, y};
        } else {
            auto [x, y, z] = morton::decode_3d(code);
            return {x, y, z};
        }
    }

    /**
     * @brief Encodes integer coordinates into a Morton code.
     */
    [[nodiscard]] uint64_t encode(const Point& p) const {
        if constexpr (DIM == 2) return morton::encode_2d(static_cast<uint32_t>(p[0]), static_cast<uint32_t>(p[1]));
        else return morton::encode_3d(static_cast<uint32_t>(p[0]), static_cast<uint32_t>(p[1]), static_cast<uint32_t>(p[2]));
    }

    /**
     * @brief Computes the Morton code of a neighbor in a given direction.
     * * Used extensively in 2:1 Balance and Ghost Layer creation.
     * Returns UINT64_MAX if the neighbor is outside the domain boundary.
     */
    [[nodiscard]] uint64_t get_neighbor_code(uint64_t code, int level, const std::array<int, DIM>& dir) const {
        auto coords = decode(code);
        uint64_t size = 1ULL << (max_level - level);
        int64_t limit = 1ULL << max_level;

        for (int i = 0; i < DIM; ++i) {
            int64_t val = static_cast<int64_t>(coords[i]) + dir[i] * static_cast<int64_t>(size);
            if (val < 0 || val >= limit) return UINT64_MAX; // Boundary
            coords[i] = static_cast<uint64_t>(val);
        }
        return encode(coords);
    }

    // --- High-Level Algorithms (Burstedde §3, Holke §4.4) ---

    /**
     * @brief Refine: Refines mesh based on Oracle.
     * * Implements parallel refinement using a parallel prefix sum (scan).
     * This avoids locks by pre-calculating the write offset for each thread.
     * * @param oracle Function object returning true if a node should be refined.
     * @return true if any nodes were refined, false otherwise.
     */
    template <RefinementOracle Oracle>
    bool refine(const Oracle& oracle) {
        bool changed = false;
        size_t n = leaves.size();
        std::vector<uint64_t> counts(n);
        std::vector<uint8_t> decisions(n);

        // 1. Decision Phase: Evaluate Oracle for all leaves in parallel
        #pragma omp parallel for reduction(|:changed)
        for (size_t i = 0; i < n; ++i) {
            // Check refinement criteria
            if (oracle(leaves[i], max_level)) {
                decisions[i] = 1;
                counts[i] = (1ULL << DIM); // Will be replaced by 4 (2D) or 8 (3D) children
                changed = true;
            } else {
                decisions[i] = 0;
                counts[i] = 1; // Kept as is
            }
        }

        if (!changed) return false;

        // 2. Scan Phase: Parallel Exclusive Scan to determine write offsets
        std::vector<uint64_t> offsets(n);
        parallel::exclusive_scan(counts, offsets);
        
        // Allocate exact size for new leaves
        std::vector<Node> next_leaves(offsets.back() + counts.back());

        // Precompute child coordinate offsets (0,0), (1,0), (0,1)...
        std::vector<Point> child_deltas;
        int num_children = 1 << DIM;
        for (int i = 0; i < num_children; ++i) {
            Point p;
            for (int d = 0; d < DIM; ++d) p[d] = (i >> d) & 1;
            child_deltas.push_back(p);
        }

        // 3. Construction Phase: Generate new leaves in parallel
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            size_t pos = offsets[i];
            if (decisions[i]) {
                const auto& node = leaves[i];
                auto coords = decode(node.code);
                int new_lvl = node.level + 1;
                uint64_t step = 1ULL << (max_level - new_lvl);

                for (int k = 0; k < num_children; ++k) {
                    Point c = coords;
                    for (int d = 0; d < DIM; ++d) c[d] += child_deltas[k][d] * step;
                    next_leaves[pos + k] = {encode(c), new_lvl};
                }
            } else {
                // Copy existing node
                next_leaves[pos] = leaves[i];
            }
        }

        leaves = std::move(next_leaves);
        return true;
    }

    /**
     * @brief Coarsen: Merges families of nodes into their parent.
     * * A family of nodes (siblings) is merged if:
     * 1. All siblings are present in the current mesh (valid family).
     * 2. The oracle returns false for the parent (indicating it doesn't need refinement).
     */
    template <RefinementOracle Oracle>
    bool coarsen(const Oracle& oracle) {
        size_t n = leaves.size();
        if (n == 0) return false;
        
        constexpr int siblings = 1 << DIM;
        std::vector<uint8_t> action(n, 0); // 0=Keep, 1=MergeToParent, 2=Delete
        bool changed = false;

        // Iterate by blocks of siblings
        #pragma omp parallel for schedule(static) reduction(|:changed)
        for (size_t i = 0; i < n; i += siblings) {
            if (i + siblings > n) continue; // Boundary check

            const auto& first = leaves[i];
            int lvl = first.level;

            if (lvl == 0) continue; // Root cannot be coarsened

            // Check if this contiguous block forms a valid sibling family
            bool valid_family = true;
            uint64_t size = 1ULL << (max_level - lvl);
            uint64_t parent_size = size * 2;
            
            // Check alignment (anchor of first child must align with parent anchor)
            auto coords = decode(first.code);
            for(auto c : coords) if (c % parent_size != 0) { valid_family = false; break; }

            if (valid_family) {
                // Check if all subsequent nodes are siblings (same level)
                for (int k = 1; k < siblings; ++k) {
                    if (leaves[i+k].level != lvl) { valid_family = false; break; }
                }
            }

            // Consult Oracle
            if (valid_family) {
                Node parent = {first.code, lvl - 1};
                // If oracle returns false, it means "Parent does NOT need refinement", so we can coarsen.
                if (!oracle(parent, max_level)) {
                    action[i] = 1; // This node becomes the parent
                    for(int k=1; k<siblings; ++k) action[i+k] = 2; // These nodes are removed
                    changed = true;
                }
            }
        }

        if (!changed) return false;

        // Stream compaction using prefix sum
        std::vector<uint64_t> keep_mask(n);
        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) keep_mask[i] = (action[i] != 2) ? 1 : 0;

        std::vector<uint64_t> offsets(n);
        parallel::exclusive_scan(keep_mask, offsets);

        std::vector<Node> next_leaves(offsets.back() + keep_mask.back());

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            if (action[i] == 2) continue;
            size_t pos = offsets[i];
            if (action[i] == 1) {
                next_leaves[pos] = {leaves[i].code, leaves[i].level - 1};
            } else {
                next_leaves[pos] = leaves[i];
            }
        }

        leaves = std::move(next_leaves);
        return true;
    }

    /**
     * @brief Balance: Enforces 2:1 Constraint (Ripple Algorithm).
     * * Refer to Holke §8.2: "The Ripple-balance algorithm".
     * Iteratively refines elements that violate the level difference condition.
     */
    void balance() {
        std::vector<std::array<int, DIM>> dirs;
        if constexpr (DIM == 2) dirs = {{1,0}, {-1,0}, {0,1}, {0,-1}};
        else dirs = {{1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1}};

        while (true) {
            size_t n = leaves.size();
            std::vector<uint8_t> refine_flags(n, 0);
            bool violation = false;

            #pragma omp parallel for schedule(dynamic, 512) reduction(|:violation)
            for (size_t i = 0; i < n; ++i) {
                const auto& node = leaves[i];
                
                for (const auto& dir : dirs) {
                    uint64_t n_code_base = get_neighbor_code(node.code, node.level, dir);
                    if (n_code_base == UINT64_MAX) continue; // Boundary

                    // Search for the neighbor in the linear tree.
                    // We must check if the space adjacent to 'node' is occupied by a 
                    // neighbor that is TOO COARSE (level < node.level - 1).
                    // We check all possible coarse levels starting from (node.level - 2).
                    int search_lvl = node.level - 2; 
                    
                    while (search_lvl >= 0) {
                        int shift = (max_level - search_lvl) * DIM;
                        uint64_t mask = (shift >= 64) ? 0 : (~0ULL << shift);
                        uint64_t target_code = n_code_base & mask;

                        // Binary search for this specific coarse code
                        auto it = std::lower_bound(leaves.begin(), leaves.end(), Node{target_code, 0});
                        
                        if (it != leaves.end() && it->code == target_code) {
                            // Found a node covering the neighbor's space.
                            
                            // If the found neighbor is indeed coarser or equal to search_lvl,
                            // it violates the 2:1 constraint relative to 'node'.
                            // (Condition: |node.level - neighbor.level| <= 1)
                            // Here: neighbor.level <= node.level - 2.
                            if (it->level <= search_lvl) {
                                size_t idx = std::distance(leaves.begin(), it);
                                // Mark the NEIGHBOR for refinement
                                #pragma omp atomic write
                                refine_flags[idx] = 1;
                                violation = true;
                                break; // Found the neighbor, stop searching levels
                            } else {
                                // The node we found is actually finer than our search level.
                                // It does not violate the condition at this search granularity.
                                break; 
                            }
                        }
                        search_lvl--;
                    }
                }
            }

            if (!violation) break;

            // Apply refinements triggered by balance violations
            refine_from_flags(refine_flags);
        }
    }

private:
    /**
     * @brief Helper to refine specific nodes marked by the balance algorithm.
     * Reuses the parallel scan logic from refine().
     */
    void refine_from_flags(const std::vector<uint8_t>& flags) {
        size_t n = leaves.size();
        std::vector<uint64_t> counts(n);
        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) counts[i] = (flags[i] ? (1ULL << DIM) : 1);

        std::vector<uint64_t> offsets(n);
        parallel::exclusive_scan(counts, offsets);

        std::vector<Node> next(offsets.back() + counts.back());
        std::vector<Point> child_deltas;
        for(int i=0; i<(1<<DIM); ++i) {
            Point p; for(int d=0; d<DIM; ++d) p[d] = (i>>d)&1;
            child_deltas.push_back(p);
        }

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            size_t pos = offsets[i];
            if (flags[i]) {
                auto coords = decode(leaves[i].code);
                int lvl = leaves[i].level + 1;
                uint64_t step = 1ULL << (max_level - lvl);
                for (int k = 0; k < (1<<DIM); ++k) {
                    Point c = coords;
                    for(int d=0; d<DIM; ++d) c[d] += child_deltas[k][d] * step;
                    next[pos+k] = {encode(c), lvl};
                }
            } else {
                next[pos] = leaves[i];
            }
        }
        leaves = std::move(next);
    }
};

using Quadtree = LinearTree<2>;
using Octree   = LinearTree<3>;

} // namespace amr