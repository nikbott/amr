/**
 * @file tests.cpp
 * @brief Validation Suite for Linear AMR System (Catch2 v3) - MPI
 * @details
 * Validates the core invariants described in AMR literature:
 * 1. **Bijection**: Morton coding must be lossless and reversible.
 * 2. **Partition of Unity**: The sum of leaf volumes must exactly equal the domain volume.
 * 3. **2:1 Balance**: Verified using an independent geometric search (brute-force).
 */

#define CATCH_CONFIG_RUNNER
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include <catch2/catch_session.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <mpi.h>

#include "core.hpp"
#include "physics.hpp"
#include "tree.hpp"

using namespace amr;
using namespace Catch::Matchers;

// ==================================================================================
// HELPER: Robust Balance Verification (MPI Version)
// ==================================================================================

/**
 * @brief Counts 2:1 Balance Violations using independent geometric verification.
 */
template <int DIM>
int count_balance_violations(const DistributedTree<DIM>& tree) {
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
    int missing_neighbors = 0;  // New: Track holes in the domain

    // Domain width in integer coordinates
    int64_t width = static_cast<int64_t>(tree.domain_width());
    const auto& ghosts = tree.get_ghosts();

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
            typename DistributedTree<DIM>::Point target_pt;
            for (int k = 0; k < DIM; ++k)
                target_pt[k] = {static_cast<uint32_t>(target_pos[k])};

            MortonCode target_code;
            if constexpr (DIM == 2)
                target_code = morton::encode_2d(target_pt[0], target_pt[1]);
            else
                target_code = morton::encode_3d(target_pt[0], target_pt[1], target_pt[2]);

            // 3. Search for the leaf covering 'target_code'
            Node search_key{target_code, 0};
            int neighbor_level = -100;  // Sentinel

            // Check Local
            auto it = std::lower_bound(
                tree.begin(), tree.end(), search_key, [](const Node& a, const Node& b) {
                    return a.code < b.code;
                });

            if (it != tree.end() && it->code == target_code) {
                neighbor_level = it->level;
            } else if (it != tree.begin()) {
                auto prev_it = it - 1;
                uint64_t prev_size = 1ULL << (DIM * (tree.max_level - prev_it->level));
                if (prev_it->code.value <= target_code.value &&
                    (prev_it->code.value + prev_size) > target_code.value) {
                    neighbor_level = prev_it->level;
                }
            }

            // Check Ghosts (if not found locally)
            if (neighbor_level == -100) {
                auto git = std::lower_bound(
                    ghosts.begin(), ghosts.end(), search_key, [](const Node& a, const Node& b) {
                        return a.code < b.code;
                    });

                if (git != ghosts.end() && git->code == target_code) {
                    neighbor_level = git->level;
                } else if (git != ghosts.begin()) {
                    auto prev_git = git - 1;
                    uint64_t prev_size = 1ULL << (DIM * (tree.max_level - prev_git->level));
                    if (prev_git->code.value <= target_code.value &&
                        (prev_git->code.value + prev_size) > target_code.value) {
                        neighbor_level = prev_git->level;
                    }
                }
            }

            // 4. Check 2:1 constraint
            if (neighbor_level == -100) {
                missing_neighbors++;  // FAIL: Expected a neighbor but found none (Hole)
            } else {
                // If neighbor is strictly coarser than (node.level - 1), violation.
                if (neighbor_level < node.level - 1) {
                    violations++;
                }
            }
        }
    }

    // Sum violations and missing neighbors across all ranks
    int global_violations = 0;
    int global_missing = 0;
    MPI_Allreduce(&violations, &global_violations, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&missing_neighbors, &global_missing, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    return global_violations + global_missing;
}

// ==================================================================================
// TEST SUITE
// ==================================================================================

TEST_CASE("Space-Filling Curve Bijection (Holke §3.1)", "[core][sfc]") {
    // Deterministic Random Generator (Same seed as original)
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint32_t> dist(0, std::numeric_limits<uint32_t>::max());

    SECTION("2D Morton Encoding (Strong Types)") {
        for (int i = 0; i < 10000; ++i) {  // Restored to 10000
            // Mask to 31 bits
            Coordinate x{dist(rng) & 0x7FFFFFFF};
            Coordinate y{dist(rng) & 0x7FFFFFFF};

            MortonCode code = morton::encode_2d(x, y);
            auto [dx, dy] = morton::decode_2d(code);

            CHECK(x.value == dx.value);
            CHECK(y.value == dy.value);
        }
    }

    SECTION("3D Morton Encoding (Strong Types)") {
        for (int i = 0; i < 10000; ++i) {  // Restored to 10000
            // Mask to 21 bits
            Coordinate x{dist(rng) & 0x1FFFFF};
            Coordinate y{dist(rng) & 0x1FFFFF};
            Coordinate z{dist(rng) & 0x1FFFFF};

            MortonCode code = morton::encode_3d(x, y, z);
            auto [dx, dy, dz] = morton::decode_3d(code);

            CHECK(x.value == dx.value);
            CHECK(y.value == dy.value);
            CHECK(z.value == dz.value);
        }
    }
}

TEMPLATE_TEST_CASE("Linear Tree Invariants (Burstedde §2.2)",
                   "[tree]",
                   DistributedTree<2>,
                   DistributedTree<3>) {
    constexpr int DIM = (std::is_same<TestType, DistributedTree<2>>::value) ? 2 : 3;

    SECTION("Completeness & Partition of Unity") {
        int max_lvl = 5;
        TestType tree(max_lvl);
        std::mt19937 rng(123);  // Same seed

        // Create a non-uniform random mesh
        tree.refine([&](const Node& n, int) { return (n.level < 4 && (rng() % 3 == 0)); });

        tree.repartition();

        // 1. Sortedness
        REQUIRE(std::is_sorted(tree.begin(), tree.end(), [](const Node& a, const Node& b) {
            return a.code < b.code;
        }));

        // 2. Uniqueness (No Duplicate Morton Codes)
        auto it = std::adjacent_find(tree.begin(), tree.end(), [](const Node& a, const Node& b) {
            return a.code == b.code;
        });
        REQUIRE(it == tree.end());

        // 3. Partition of Unity (Explicit Calculation matches original)
        double local_volume = 0.0;
        for (const auto& node : tree) {
            double side = 1.0 / (1ULL << node.level);
            local_volume += std::pow(side, DIM);
        }

        double total_volume = 0.0;
        MPI_Allreduce(&local_volume, &total_volume, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        REQUIRE_THAT(total_volume, WithinAbs(1.0, 1e-9));

        // 4. Design-By-Contract Verification
        REQUIRE_NOTHROW(tree.verify_global());
    }

    SECTION("Neighbor Query Integrity") {
        TestType tree(4);
        // Uniform refinement to L2
        while (tree.refine([&](const Node& n, int) { return n.level < 2; }))
            ;

        tree.repartition();

        // Check internal neighbor (Center of Domain)
        Coordinate size{static_cast<uint32_t>(tree.domain_width() / 4)};
        typename TestType::Point coords;
        for (int k = 0; k < DIM; ++k)
            coords[k] = size;

        MortonCode code;
        if constexpr (DIM == 2)
            code = morton::encode_2d(coords[0], coords[1]);
        else
            code = morton::encode_3d(coords[0], coords[1], coords[2]);

        // Check Neighbor to the Right (+X)
        std::array<int, DIM> dir_right = {0};
        dir_right[0] = 1;

        MortonCode right_code = tree.get_neighbor_code(code, 2, dir_right);
        REQUIRE(right_code.value != UINT64_MAX);

        auto right_coords = tree.decode(right_code);
        CHECK(right_coords[0].value == coords[0].value + size.value);
    }
}

TEMPLATE_TEST_CASE("Adaptivity & Coarsening (Burstedde §3.2)",
                   "[amr][coarsen]",
                   DistributedTree<2>,
                   DistributedTree<3>) {
    constexpr int DIM = (std::is_same<TestType, DistributedTree<2>>::value) ? 2 : 3;

    SECTION("Geometric Coarsening") {
        TestType tree(6);
        // Refine Root -> L1 (Leaves = 2^DIM)
        tree.refine([&](const Node& n, int) { return n.level < 1; });

        // Check Global Size
        size_t n_global = tree.global_size();
        REQUIRE(n_global == (1ULL << DIM));

        // Coarsen L1 -> Root
        bool changed = tree.coarsen([&](const Node&, int) { return false; });

        REQUIRE(changed);
        REQUIRE(tree.global_size() == 1);

        // Root check (only valid on the rank that owns root, but global size 1 implies strictness)
        if (tree.local_size() > 0) {
            CHECK(tree[0].level == 0);
        }
        REQUIRE_NOTHROW(tree.verify_global());
    }

    SECTION("Graded Coarsening (Misaligned Families)") {
        // Test robustness against misaligned families in the linear array.
        TestType tree(5);

        // Create Graded Mesh: Refine all to L1, then refine the LAST L1 node to L2.
        int steps = 0;
        while (tree.refine([&](const Node& n, int) {
            if (n.level == 0)
                return true;
            if (n.level == 1) {
                // Find the last sibling (Code ends in 11...1)
                uint64_t last_sib_idx = (1ULL << DIM) - 1;
                uint64_t shift = DIM * (tree.max_level - 1);
                return ((n.code.value >> shift) & last_sib_idx) == last_sib_idx;
            }
            return false;
        }) && ++steps < 10)
            ;

        // Expectation:
        // L1 Nodes: (2^DIM - 1)
        // L2 Nodes: (2^DIM) (Refined from the last L1)
        size_t expected_size = ((1ULL << DIM) - 1) + (1ULL << DIM);
        REQUIRE(tree.global_size() == expected_size);
        REQUIRE_NOTHROW(tree.verify_global());

        // Force coarsening.
        // The L2 family should merge. The misaligned L1 nodes should eventually merge.
        int passes = 0;
        while (tree.coarsen([&](const Node&, int) { return false; })) {
            passes++;
            tree.verify_global();
        }

        // Should return to root
        REQUIRE(tree.global_size() == 1);
        if (tree.local_size() > 0) {
            CHECK(tree[0].level == 0);
        }
    }
}

TEMPLATE_TEST_CASE("2:1 Balance & Ripple Algorithm (Holke §3.3)",
                   "[balance]",
                   DistributedTree<2>,
                   DistributedTree<3>) {
    constexpr int DIM = (std::is_same<TestType, DistributedTree<2>>::value) ? 2 : 3;

    SECTION("Deep Ripple (Cascade Propagation)") {
        // "Tower" Test: Refine center to Max Level.
        int max_lvl = 6;
        TestType tree(max_lvl);

        auto center_oracle = [&](const Node& n, int) {
            auto coords = tree.decode(n.code);
            uint64_t size = 1ULL << (max_lvl - n.level);
            uint64_t mid = tree.domain_width() / 2;
            bool contains_center = true;
            for (int k = 0; k < DIM; ++k) {
                if (!(coords[k].value <= mid && coords[k].value + size > mid))
                    contains_center = false;
            }
            return (contains_center && n.level < max_lvl);
        };

        // To match original:
        while (tree.refine(center_oracle))
            ;

        tree.repartition();

        // Execute Balance
        tree.balance();
        tree.verify_global();

        // Verification
        int violations = count_balance_violations(tree);
        REQUIRE(violations == 0);
    }

    SECTION("Random Cloud Stress Test") {
        TestType tree(8);
        auto cloud_oracle = [&](const Node& n, int) {
            if (n.level >= 5)
                return false;
            return (n.code.value % 7 == 0 || n.code.value % 13 == 0);
        };

        for (int i = 0; i < 8; ++i)
            tree.refine(cloud_oracle);

        tree.repartition();

        tree.balance();
        tree.verify_global();

        int violations = count_balance_violations(tree);
        REQUIRE(violations == 0);
    }
}

TEST_CASE("Safety & Edge Cases", "[safety]") {
    // Only need one dimension type to test general logic
    using TestType = DistributedTree<2>;

    SECTION("Empty Tree Violation") {
        TestType tree(5);
        REQUIRE(tree.global_size() > 0);
        REQUIRE_NOTHROW(tree.verify_global());
    }

    SECTION("Max Level Constraints") {
        // 2D Max is 31
        REQUIRE_NOTHROW(DistributedTree<2>(31));
        // 3D Max is 21
        REQUIRE_NOTHROW(DistributedTree<3>(21));
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int result = Catch::Session().run(argc, argv);
    MPI_Finalize();
    return result;
}
