/**
 * @file test_util.hpp
 * @brief Independent oracles shared by the OpenMP backend's test files.
 *
 * @details The suite (omp/test_*.cpp) checks the invariants of the AMR
 * literature: Morton coding is a bijection, leaf volumes partition the domain
 * exactly, and 2:1 balance holds, counted here by an independent geometric
 * search rather than by the optimized balance itself.
 */
#pragma once

#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core.hpp"
#include "tree.hpp"

namespace amr::test {

using namespace amr;

/**
 * @brief Counts 2:1 Balance Violations using independent geometric verification.
 */
template <int DIM>
int count_balance_violations(const LinearTree<DIM>& tree) {
    // Full 2:1 balance: probe the 12 edge diagonals too, or edge violations go
    // uncounted (the whole point of full balance for the DIC bridge).
    std::vector<std::array<int, DIM>> dirs;
    if constexpr (DIM == 2)
        dirs = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    else
        dirs = {{1, 0, 0},
                {-1, 0, 0},
                {0, 1, 0},
                {0, -1, 0},
                {0, 0, 1},
                {0, 0, -1},
                {1, 1, 0},
                {1, -1, 0},
                {-1, 1, 0},
                {-1, -1, 0},
                {1, 0, 1},
                {1, 0, -1},
                {-1, 0, 1},
                {-1, 0, -1},
                {0, 1, 1},
                {0, 1, -1},
                {0, -1, 1},
                {0, -1, -1}};

    int violations = 0;
    int missing_neighbors = 0;  // New: Track holes

    // Domain width in integer coordinates
    int64_t width = static_cast<int64_t>(tree.domain_width());

    for (const auto& node : tree) {
        // Independent decoding
        auto coords = tree.decode(node.code);
        int64_t size = 1LL << (tree.max_level - node.level);

        for (const auto& dir : dirs) {
            // 1. Calculate Expected Geometric Neighbor Coordinate
            std::array<int64_t, DIM> target_pos;
            bool out_of_bounds = false;

            for (int k = 0; k < DIM; ++k) {
                int64_t current_val = static_cast<int64_t>(coords[k].value);
                if (dir[k] == 1) {
                    target_pos[k] = current_val + size;
                } else if (dir[k] == -1) {
                    target_pos[k] = current_val - 1;
                } else {
                    target_pos[k] = current_val;
                }

                // Bounds Check
                if (target_pos[k] < 0 || target_pos[k] >= width) {
                    out_of_bounds = true;
                    break;
                }
            }
            if (out_of_bounds)
                continue;

            // 2. Re-encode to find the code covering this point
            typename LinearTree<DIM>::Point target_pt;
            for (int k = 0; k < DIM; ++k)
                target_pt[k] = {static_cast<uint32_t>(target_pos[k])};

            MortonCode target_code = tree.encode(target_pt);

            // 3. Search for the leaf covering 'target_code'
            // We search for target_code. The covering leaf will be <= target_code.
            auto it = std::lower_bound(tree.begin(), tree.end(), Node{target_code, 0});

            Node neighbor{MortonCode{0}, 0};
            bool found = false;

            if (it != tree.end() && it->code == target_code) {
                // Exact match (rare, but possible if anchor aligns)
                neighbor = *it;
                found = true;
            } else if (it != tree.begin()) {
                // Check predecessor
                auto prev_it = it - 1;
                Node prev = *prev_it;

                uint64_t prev_size = 1ULL << (DIM * (tree.max_level - prev.level));
                if (prev.code.value <= target_code.value &&
                    (prev.code.value + prev_size) > target_code.value) {
                    neighbor = prev;
                    found = true;
                }
            }

            // 4. Check 2:1 constraint (STRICT)
            if (!found) {
                missing_neighbors++;  // FAIL: Hole in domain
            } else {
                // If neighbor is strictly coarser than (node.level - 1), violation.
                if (neighbor.level < node.level - 1) {
                    violations++;
                }
            }
        }
    }
    return violations + missing_neighbors;
}

// Leaf set as (code, level) pairs in Morton order.
template <int DIM>
std::vector<std::pair<uint64_t, int>> leaves_of(const LinearTree<DIM>& tree) {
    std::vector<std::pair<uint64_t, int>> out;
    for (const auto& n : tree)
        out.emplace_back(n.code.value, n.level);
    return out;
}

// True when every leaf of `after` lies inside a leaf of `before` at the same or
// a coarser level: balance may only refine, never coarsen or move leaves.
template <int DIM>
bool only_refines(const std::vector<std::pair<uint64_t, int>>& before,
                  const std::vector<std::pair<uint64_t, int>>& after,
                  int max_level) {
    for (const auto& [code, level] : after) {
        auto it = std::upper_bound(before.begin(), before.end(), std::make_pair(code, INT32_MAX));
        if (it == before.begin())
            return false;
        --it;
        const uint64_t span = 1ULL << (DIM * (max_level - it->second));
        if (code < it->first || code >= it->first + span || it->second > level)
            return false;
    }
    return true;
}

// Balance must fix real violations, only refine, and be idempotent.
template <int DIM>
void check_balance_properties(LinearTree<DIM>& tree) {
    REQUIRE(count_balance_violations(tree) > 0);  // fixture is not vacuous
    const auto before = leaves_of(tree);
    tree.balance();
    tree.verify();
    REQUIRE(count_balance_violations(tree) == 0);
    const auto after = leaves_of(tree);
    REQUIRE(only_refines<DIM>(before, after, tree.max_level));
    tree.balance();
    REQUIRE(leaves_of(tree) == after);  // idempotent
}

}  // namespace amr::test
