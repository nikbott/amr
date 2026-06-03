/**
 * @file tree.hpp
 * @brief Linear Quadtree/Octree Implementation.
 * * @details 
 * Implements a **Linear Tree** [Burstedde 2011], where the quadtree/octree is represented 
 * not by pointers, but by a sorted vector of leaf nodes (Morton codes).
 * * Features:
 * - **Structure-of-Arrays (SoA)**: Stores codes and levels in separate vectors 
 * to maximize cache line utilization during traversals.
 * - **Ripple Balance**: Implements the 2:1 balance constraint (adjacent elements 
 * differ by at most 1 refinement level) using an iterative "ripple" propagation 
 * algorithm [Holke 2018].
 * - **Pointer-less Neighbor Finding**: Uses bitwise arithmetic to calculate 
 * neighbor codes, avoiding tree traversal overhead.
 */

#pragma once
#include "core.hpp"
#include <vector>
#include <array>
#include <cassert>
#include <omp.h>
#include <algorithm>
#include <iterator>
#include <memory>
#include <memory_resource>
#include <stdexcept>

namespace amr {

struct Node {
    MortonCode code;
    int level;
    // Comparisons for sorting and searching
    bool operator<(const Node& other) const { return code < other.code; }
    bool operator<(const uint64_t& val) const { return code.value < val; }
    friend bool operator<(const uint64_t& val, const Node& n) { return val < n.code.value; }
    bool operator==(const Node& other) const { return code == other.code && level == other.level; }
};

template<typename T>
concept RefinementOracle = requires(T t, const Node& n, int max_lvl) {
    { t(n, max_lvl) } -> std::convertible_to<bool>;
};

template <int DIM>
class LinearTree {
    static_assert(DIM == 2 || DIM == 3, "Only 2D or 3D trees supported.");
    static_assert((DIM == 3 && sizeof(uint64_t) * 8 >= 63) || (DIM == 2), 
                  "64-bit integers required for 21-level Octree");

public:
    using Point = std::array<Coordinate, DIM>;
    const int max_level;
    static constexpr int dim = DIM;

private:
    // SoA Tree Storage: Separating hot/cold data for better SIMD/Cache usage
    std::vector<Uninit<uint64_t>> leaf_codes;
    std::vector<Uninit<uint8_t>>  leaf_levels;
    
    // Double buffering for parallel atomic updates
    std::vector<Uninit<uint64_t>> buffer_codes;
    std::vector<Uninit<uint8_t>>  buffer_levels;

    // Persistent Workspace Buffers (Reuse memory to reduce allocation overhead)
    std::vector<Uninit<uint64_t>> wksp_counts;
    std::vector<Uninit<uint64_t>> wksp_offsets;
    std::vector<uint8_t>  wksp_flags;
    std::vector<Uninit<uint64_t>> wksp_mask;
    std::vector<uint64_t> wksp_scan_buffer;

public:
    // --- ITERATOR ---
    // Standard Random Access Iterator to allow usage with std::algorithms
    class ConstIterator {
        const LinearTree* tree;
        size_t index;
    public:
        friend class LinearTree;
        using iterator_concept  = std::random_access_iterator_tag;
        using iterator_category = std::random_access_iterator_tag;
        using value_type        = Node;
        using difference_type   = std::ptrdiff_t;
        using pointer           = void; 
        using reference         = Node;

        ConstIterator(const LinearTree* t, size_t i) : tree(t), index(i) {}

        reference operator*() const {
            return { MortonCode{tree->leaf_codes[index]}, static_cast<int>(tree->leaf_levels[index]) };
        }

        struct ArrowProxy {
            Node value;
            const Node* operator->() const { return &value; }
        };
        ArrowProxy operator->() const { return ArrowProxy{ **this }; }

        MortonCode code() const { return {tree->leaf_codes[index]}; }
        int level() const { return static_cast<int>(tree->leaf_levels[index]); }

        ConstIterator& operator++() { ++index; return *this; }
        ConstIterator operator++(int) { ConstIterator tmp = *this; ++index; return tmp; }
        ConstIterator& operator--() { --index; return *this; }
        ConstIterator operator--(int) { ConstIterator tmp = *this; --index; return tmp; }
        ConstIterator& operator+=(difference_type n) { index += n; return *this; }
        ConstIterator& operator-=(difference_type n) { index -= n; return *this; }
        reference operator[](difference_type n) const { return *(*this + n); }
        
        friend ConstIterator operator+(ConstIterator it, difference_type n) { return {it.tree, it.index + n}; }
        friend ConstIterator operator+(difference_type n, ConstIterator it) { return {it.tree, it.index + n}; }
        friend ConstIterator operator-(ConstIterator it, difference_type n) { return {it.tree, it.index - n}; }
        friend difference_type operator-(const ConstIterator& a, const ConstIterator& b) { 
            return static_cast<difference_type>(a.index) - static_cast<difference_type>(b.index); 
        }
        auto operator<=>(const ConstIterator& other) const = default;
    };

    explicit LinearTree(int max_lvl) 
        : max_level(max_lvl)
    {
        leaf_codes.push_back(0);
        leaf_levels.push_back(0);
        buffer_codes.reserve(1024);
        buffer_levels.reserve(1024);
    }

    LinearTree(LinearTree&&) = default;
    LinearTree& operator=(LinearTree&&) = default;
    LinearTree(const LinearTree&) = delete;
    LinearTree& operator=(const LinearTree&) = delete;

    // --- Public API ---
    ConstIterator begin() const { return ConstIterator(this, 0); }
    ConstIterator end() const { return ConstIterator(this, leaf_codes.size()); }
    Node operator[](size_t i) const { return {MortonCode{leaf_codes[i]}, static_cast<int>(leaf_levels[i])}; }
    [[nodiscard]] size_t size() const { return leaf_codes.size(); }
    [[nodiscard]] uint64_t domain_width() const { return 1ULL << max_level; }

    [[nodiscard]] Point decode(MortonCode code) const {
        if constexpr (DIM == 2) return morton::decode_2d(code);
        else return morton::decode_3d(code);
    }

    [[nodiscard]] MortonCode encode(const Point& p) const {
        if constexpr (DIM == 2) return morton::encode_2d(p[0], p[1]);
        else return morton::encode_3d(p[0], p[1], p[2]);
    }

    /**
     * @brief Computes the Morton code of a neighbor using bitwise arithmetic.
     * @details 
     * Unlike pointer-based trees, Linear Octrees allow finding neighbors via integer operations
     * on the Morton code, which is significantly faster.
     * * @param code The Morton code of the current node.
     * @param level The level of the current node.
     * @param dir The direction vector (e.g., {1,0} for +X).
     * @return The neighbor's code (aligned to the same level), or UINT64_MAX on boundary overflow.
     */
    [[nodiscard]] MortonCode get_neighbor_code(MortonCode code, int level, const std::array<int, DIM>& dir) const {
        // Integer-Intrinsic Neighbor Finding (Bitwise Arithmetic)
        uint64_t c = code.value;
        uint64_t mask_x, mask_y, mask_z;
        
        if constexpr (DIM == 3) {
            mask_x = morton::MASK3_X; mask_y = morton::MASK3_Y; mask_z = morton::MASK3_Z;
        } else {
            mask_x = morton::MASK2_X; mask_y = 0xAAAAAAAAAAAAAAAA; mask_z = 0;
        }

        auto add_dim = [&](uint64_t current, uint64_t dim_mask, int d) -> uint64_t {
            if (current == UINT64_MAX) return UINT64_MAX; // Propagate error
            if (d == 0) return current;
            
            uint64_t shift_coord = max_level - level;
            uint64_t one_dilated;
            
            if constexpr (DIM == 2) {
                 one_dilated = 1ULL << (shift_coord * 2); 
                 if (dim_mask == mask_y) one_dilated <<= 1;
            } else {
                 one_dilated = 1ULL << (shift_coord * 3);
                 if (dim_mask == mask_y) one_dilated <<= 1;
                 if (dim_mask == mask_z) one_dilated <<= 2;
            }

            if (d > 0) {
                 // Boundary Check: If setting all 'holes' to 1 creates ALL_ONES, we overflow
                 if ((current | ~dim_mask) == UINT64_MAX) return UINT64_MAX;
                 
                 uint64_t sum = (current | ~dim_mask) + one_dilated;
                 return (sum & dim_mask) | (current & ~dim_mask);
            } else {
                 // Boundary Check: If value in dimension is less than step, we underflow
                 if ((current & dim_mask) < one_dilated) return UINT64_MAX;

                 uint64_t diff = (current & dim_mask) - one_dilated;
                 return (diff & dim_mask) | (current & ~dim_mask);
            }
        };

        uint64_t next = c;
        next = add_dim(next, mask_x, dir[0]);
        next = add_dim(next, mask_y, dir[1]);
        if constexpr (DIM == 3) {
            next = add_dim(next, mask_z, dir[2]);
        }
        return MortonCode{next};
    }

    /**
     * @brief Verifies the "Partition of Unity" and "Sortedness" invariants.
     * @throws std::runtime_error if the tree is invalid.
     */
    void verify() const {
        if (leaf_codes.empty()) throw std::runtime_error("Empty tree violation");
        if (!std::is_sorted(leaf_codes.begin(), leaf_codes.end())) throw std::runtime_error("Leaves not sorted");

        double total_volume = 0.0;
        uint64_t last_end = 0;
        for (size_t i = 0; i < size(); ++i) {
            uint64_t code = leaf_codes[i];
            int lvl = leaf_levels[i];
            if (code < last_end) throw std::runtime_error("Overlapping nodes detected");
            double vol = 1.0 / (1ULL << lvl);
            total_volume += std::pow(vol, DIM);
            last_end = code + (1ULL << (DIM * (max_level - lvl)));
        }
        if (std::abs(total_volume - 1.0) > 1e-9) throw std::runtime_error("Volume sum != 1.0");
    }

    template <RefinementOracle Oracle>
    bool refine(const Oracle& oracle) {
        bool changed = false;
        size_t n = leaf_codes.size();
        wksp_counts.resize(n);
        wksp_offsets.resize(n);

        #pragma omp parallel for reduction(|:changed)
        for (size_t i = 0; i < n; ++i) {
            int lvl = static_cast<int>(leaf_levels[i]);
            if (lvl < max_level && oracle({MortonCode{leaf_codes[i]}, lvl}, max_level)) {
                wksp_counts[i] = (1ULL << DIM); 
                changed = true;
            } else {
                wksp_counts[i] = 1; 
            }
        }
        if (!changed) return false;

        parallel::exclusive_scan(wksp_counts, wksp_offsets, wksp_scan_buffer);
        
        size_t total_new_nodes = wksp_offsets.back() + wksp_counts.back();
        buffer_codes.clear(); buffer_codes.resize(total_new_nodes);
        buffer_levels.clear(); buffer_levels.resize(total_new_nodes);

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            size_t pos = wksp_offsets[i];
            uint64_t code = leaf_codes[i];
            int lvl = leaf_levels[i];

            if (wksp_counts[i] > 1) {
                int new_lvl = lvl + 1;
                uint64_t shift = (max_level - new_lvl) * DIM;
                for (int k = 0; k < (1<<DIM); ++k) {
                    uint64_t child_code = code | (static_cast<uint64_t>(k) << shift);
                    buffer_codes[pos + k] = child_code;
                    buffer_levels[pos + k] = static_cast<uint8_t>(new_lvl);
                }
            } else {
                buffer_codes[pos] = code;
                buffer_levels[pos] = static_cast<uint8_t>(lvl);
            }
        }
        leaf_codes.swap(buffer_codes);
        leaf_levels.swap(buffer_levels);
        return true;
    }

    template <RefinementOracle Oracle>
    bool coarsen(const Oracle& oracle) {
        size_t n = leaf_codes.size();
        if (n == 0) return false;
        
        constexpr int siblings = 1 << DIM;
        wksp_flags.assign(n, 0); 
        bool changed = false;

        #pragma omp parallel for schedule(static) reduction(|:changed)
        for (size_t i = 0; i < n; ++i) {
            if (i + siblings > n) continue;

            uint64_t raw_code = leaf_codes[i];
            int lvl = leaf_levels[i];
            
            if (lvl == 0) continue;

            uint64_t shift = static_cast<uint64_t>(DIM) * (max_level - lvl);
            uint64_t child_idx = (raw_code >> shift) & ((1ULL << DIM) - 1);
            
            if (child_idx != 0) continue;

            bool valid_family = true;
            for (int k = 1; k < siblings; ++k) {
                if (leaf_levels[i+k] != lvl) { valid_family = false; break; }
            }

            if (valid_family) {
                if (!oracle({MortonCode{raw_code}, lvl - 1}, max_level)) {
                    wksp_flags[i] = 1; 
                    for(int k=1; k<siblings; ++k) wksp_flags[i+k] = 2; 
                    changed = true;
                }
            }
        }

        if (!changed) return false;

        wksp_mask.resize(n);
        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) wksp_mask[i] = (wksp_flags[i] != 2) ? 1 : 0;

        wksp_offsets.resize(n);
        parallel::exclusive_scan(wksp_mask, wksp_offsets, wksp_scan_buffer);

        size_t new_size = wksp_offsets.back() + wksp_mask.back();
        buffer_codes.clear(); buffer_codes.resize(new_size);
        buffer_levels.clear(); buffer_levels.resize(new_size);

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            if (wksp_flags[i] == 2) continue;
            size_t pos = wksp_offsets[i];
            if (wksp_flags[i] == 1) {
                buffer_codes[pos] = leaf_codes[i];
                buffer_levels[pos] = static_cast<uint8_t>(leaf_levels[i] - 1);
            } else {
                buffer_codes[pos] = leaf_codes[i];
                buffer_levels[pos] = leaf_levels[i];
            }
        }
        leaf_codes.swap(buffer_codes);
        leaf_levels.swap(buffer_levels);
        return true;
    }

    // Number of ripple passes the last balance*() call performed.
    int last_balance_iters = 0;

    /**
     * @brief Active-front 2:1 balance (production).
     * @details Byte-identical output to balance_ref(), but after pass 1 only the
     * advancing refinement front is re-checked (a cell can newly violate 2:1 only
     * if a face-neighbour just got finer), so cost tracks the front, not all N.
     * Mirrors the verified CUDA backend (`cuda/tree.cuh`). [Holke2018 §3.3]
     *
     * A `dirty` mask drives the re-check. After refining cells {j} in a pass, the
     * only cells that can newly violate are the children of {j} and the
     * equal-or-finer face-neighbours of those children (range-marked from the fine
     * side) — a provably complete superset, hence identical flags. An adaptive
     * fallback (children pre-filter + dirty-count backstop -> `front_collapsed`)
     * reverts to a full check on wide fronts so it never regresses.
     */
    void balance() {
        last_balance_iters = 0;
        constexpr int siblings = 1 << DIM;
        std::vector<std::array<int, DIM>> dirs;
        if constexpr (DIM == 2) dirs = {{1,0}, {-1,0}, {0,1}, {0,-1}};
        else dirs = {{1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1}};

        std::vector<uint8_t> dirty(leaf_codes.size(), 1);   // pass 1: every leaf
        std::vector<uint8_t> child_mask, dirty_next;
        bool front_collapsed = false;

        while (true) {
            size_t n = leaf_codes.size();
            wksp_flags.assign(n, 0);
            bool violation = false;

            #pragma omp parallel for schedule(dynamic, 1024) reduction(|:violation)
            for (size_t i = 0; i < n; ++i) {
                if (!dirty[i]) continue;
                MortonCode code{leaf_codes[i]};
                int lvl = leaf_levels[i];

                for (const auto& dir : dirs) {
                    MortonCode n_code_base = get_neighbor_code(code, lvl, dir);
                    if (n_code_base.value == UINT64_MAX) continue;
                    int search_lvl = lvl - 2;
                    if (search_lvl < 0) continue;

                    int shift = (max_level - search_lvl) * DIM;
                    uint64_t mask = (shift >= 64) ? 0 : (~0ULL << shift);
                    uint64_t target_code = n_code_base.value & mask;

                    auto it = std::lower_bound(leaf_codes.begin(), leaf_codes.end(), target_code);
                    if (it != leaf_codes.end() && *it == target_code) {
                        size_t idx = std::distance(leaf_codes.begin(), it);
                        if (leaf_levels[idx] <= search_lvl) { wksp_flags[idx] = 1; violation = true; }
                    } else if (it != leaf_codes.begin()) {
                        auto prev = it - 1;
                        size_t prev_idx = std::distance(leaf_codes.begin(), prev);
                        uint64_t prev_code = *prev;
                        int prev_lvl = leaf_levels[prev_idx];
                        uint64_t size = 1ULL << (DIM * (max_level - prev_lvl));
                        if (prev_code <= target_code && (prev_code + size) > target_code && prev_lvl <= search_lvl) {
                            wksp_flags[prev_idx] = 1; violation = true;
                        }
                    }
                }
            }
            if (!violation) break;
            ++last_balance_iters;

            size_t old_n = n;
            refine_from_flags();                 // wksp_offsets: old->new starts; wksp_flags: which old were refined
            size_t new_n = leaf_codes.size();

            // Seed the next pass's dirty mask over the NEW array (active front).
            // Adaptive fallback: if the front is a large fraction, a full check is
            // cheaper than maintaining it (see cuda/tree.cuh for the rationale).
            long long refined = static_cast<long long>(new_n - old_n) / (siblings - 1);
            if (front_collapsed || refined * siblings * 64 > static_cast<long long>(new_n)) {
                dirty.assign(new_n, 1);
                continue;
            }

            // Phase A: mark just-created children (over OLD refined cells).
            child_mask.assign(new_n, 0);
            #pragma omp parallel for
            for (size_t i = 0; i < old_n; ++i) {
                if (!wksp_flags[i]) continue;
                size_t base = wksp_offsets[i];
                for (int k = 0; k < siblings; ++k) child_mask[base + k] = 1;
            }

            // Phase B: from each new child, mark its equal-or-finer face-neighbours
            // (range [n_code, n_code+child_size) on the new sorted codes).
            dirty_next = child_mask;
            #pragma omp parallel for schedule(dynamic, 1024)
            for (size_t c = 0; c < new_n; ++c) {
                if (!child_mask[c]) continue;
                MortonCode code{leaf_codes[c]};
                int lvl = leaf_levels[c];
                uint64_t my_size = 1ULL << (DIM * (max_level - lvl));
                for (const auto& dir : dirs) {
                    MortonCode nc = get_neighbor_code(code, lvl, dir);
                    if (nc.value == UINT64_MAX) continue;
                    auto lo = std::lower_bound(leaf_codes.begin(), leaf_codes.end(), nc.value);
                    auto hi = std::lower_bound(leaf_codes.begin(), leaf_codes.end(), nc.value + my_size);
                    for (auto it = lo; it != hi; ++it)
                        dirty_next[std::distance(leaf_codes.begin(), it)] = 1;
                }
            }

            // Backstop: once the front is a large fraction, stop tracking it.
            long long dirty_count = 0;
            #pragma omp parallel for reduction(+:dirty_count)
            for (size_t c = 0; c < new_n; ++c) dirty_count += dirty_next[c];
            if (dirty_count * 8 > static_cast<long long>(new_n)) front_collapsed = true;

            dirty.swap(dirty_next);
        }
    }

    /**
     * @brief Reference 2:1 balance: re-checks the whole mesh every ripple pass.
     * @details Simple and obviously correct ([Holke2018 §3.3]); kept as the parity
     * oracle and baseline for the active-front balance().
     */
    void balance_ref() {
        last_balance_iters = 0;
        std::vector<std::array<int, DIM>> dirs;
        if constexpr (DIM == 2) dirs = {{1,0}, {-1,0}, {0,1}, {0,-1}};
        else dirs = {{1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1}};

        while (true) {
            size_t n = leaf_codes.size();
            wksp_flags.assign(n, 0);
            bool violation = false;

            #pragma omp parallel for schedule(dynamic, 1024) reduction(|:violation)
            for (size_t i = 0; i < n; ++i) {
                MortonCode code{leaf_codes[i]};
                int lvl = leaf_levels[i];

                for (const auto& dir : dirs) {
                    MortonCode n_code_base = get_neighbor_code(code, lvl, dir);
                    if (n_code_base.value == UINT64_MAX) continue;
                    int search_lvl = lvl - 2;
                    if (search_lvl < 0) continue;

                    int shift = (max_level - search_lvl) * DIM;
                    uint64_t mask = (shift >= 64) ? 0 : (~0ULL << shift);
                    uint64_t target_code = n_code_base.value & mask;

                    auto it = std::lower_bound(leaf_codes.begin(), leaf_codes.end(), target_code);
                    if (it != leaf_codes.end() && *it == target_code) {
                        size_t idx = std::distance(leaf_codes.begin(), it);
                        if (leaf_levels[idx] <= search_lvl) { wksp_flags[idx] = 1; violation = true; }
                    } else if (it != leaf_codes.begin()) {
                        auto prev = it - 1;
                        size_t prev_idx = std::distance(leaf_codes.begin(), prev);
                        uint64_t prev_code = *prev;
                        int prev_lvl = leaf_levels[prev_idx];
                        uint64_t size = 1ULL << (DIM * (max_level - prev_lvl));
                        if (prev_code <= target_code && (prev_code + size) > target_code && prev_lvl <= search_lvl) {
                            wksp_flags[prev_idx] = 1; violation = true;
                        }
                    }
                }
            }
            if (!violation) break;
            ++last_balance_iters;
            refine_from_flags();
        }
    }

private:
    void refine_from_flags() {
        size_t n = leaf_codes.size();
        wksp_counts.resize(n);
        wksp_offsets.resize(n);

        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) wksp_counts[i] = (wksp_flags[i] ? (1ULL << DIM) : 1);

        parallel::exclusive_scan(wksp_counts, wksp_offsets, wksp_scan_buffer);

        size_t new_size = wksp_offsets.back() + wksp_counts.back();
        if (new_size == n) return;
        buffer_codes.clear(); buffer_codes.resize(new_size);
        buffer_levels.clear(); buffer_levels.resize(new_size);
        
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            size_t pos = wksp_offsets[i];
            uint64_t code = leaf_codes[i];
            int lvl = leaf_levels[i];

            if (wksp_flags[i]) {
                int new_lvl = lvl + 1;
                uint64_t shift = (max_level - new_lvl) * DIM;
                for (int k = 0; k < (1<<DIM); ++k) {
                    uint64_t child_code = code | (static_cast<uint64_t>(k) << shift);
                    buffer_codes[pos+k] = child_code;
                    buffer_levels[pos+k] = static_cast<uint8_t>(new_lvl);
                }
            } else {
                buffer_codes[pos] = code;
                buffer_levels[pos] = static_cast<uint8_t>(lvl);
            }
        }
        leaf_codes.swap(buffer_codes);
        leaf_levels.swap(buffer_levels);
    }
};

using Quadtree = LinearTree<2>;
using Octree   = LinearTree<3>;

} // namespace amr