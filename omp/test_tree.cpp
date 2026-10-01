/**
 * @file test_tree.cpp
 * @brief Linear-tree invariants, refinement, coarsening and 2:1 balance.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "physics.hpp"
#include "test_util.hpp"

using namespace amr;
using namespace amr::test;
using namespace Catch::Matchers;

TEMPLATE_TEST_CASE("Linear Tree Invariants (Burstedde §2.2)", "[tree]", Quadtree, Octree) {
    constexpr int DIM = (std::is_same<TestType, Quadtree>::value) ? 2 : 3;

    SECTION("Completeness & Partition of Unity") {
        int max_lvl = 5;
        TestType tree(max_lvl);
        // Create a non-uniform mesh. The predicate is a hash of the cell, not a
        // shared generator: refine() calls it from many threads at once.
        tree.refine([&](const Node& n, int) {
            return n.level < 4 &&
                   ((n.code.value ^ (uint64_t(n.level) << 58)) * 0x9e3779b97f4a7c15ULL >> 40) % 3 ==
                       0;
        });

        // 1. Sortedness
        REQUIRE(std::is_sorted(tree.begin(), tree.end()));

        // 2. Uniqueness (No Duplicate Morton Codes)
        auto it = std::adjacent_find(tree.begin(), tree.end());
        REQUIRE(it == tree.end());

        // 3. Partition of Unity
        double total_volume = 0.0;
        for (const auto& node : tree) {
            double side = 1.0 / (1ULL << node.level);
            total_volume += std::pow(side, DIM);
        }
        REQUIRE_THAT(total_volume, WithinAbs(1.0, 1e-9));

        // 4. Design-By-Contract Verification
        REQUIRE_NOTHROW(tree.verify());
    }

    SECTION("Neighbor Query Integrity") {
        TestType tree(4);
        // Uniform refinement to L2
        while (tree.refine([&](const Node& n, int) { return n.level < 2; }))
            ;

        // Check internal neighbor (Center of Domain)
        Coordinate size{static_cast<uint32_t>(tree.domain_width() / 4)};
        typename TestType::Point coords;
        for (int k = 0; k < DIM; ++k)
            coords[k] = size;

        MortonCode code = tree.encode(coords);

        // Check Neighbor to the Right (+X)
        std::array<int, DIM> dir_right = {0};
        dir_right[0] = 1;

        MortonCode right_code = tree.get_neighbor_code(code, 2, dir_right);
        REQUIRE(right_code.value != UINT64_MAX);

        auto right_coords = tree.decode(right_code);
        CHECK(right_coords[0].value == coords[0].value + size.value);
    }
}

TEMPLATE_TEST_CASE("Adaptivity & Coarsening (Burstedde §3.2)", "[amr][coarsen]", Quadtree, Octree) {
    constexpr int DIM = (std::is_same<TestType, Quadtree>::value) ? 2 : 3;

    SECTION("Geometric Coarsening") {
        TestType tree(6);
        // Refine Root -> L1 (Leaves = 2^DIM)
        tree.refine([&](const Node& n, int) { return n.level < 1; });
        REQUIRE(tree.size() == (1ULL << DIM));

        // Coarsen L1 -> Root
        bool changed = tree.coarsen([&](const Node&, int) { return false; });

        REQUIRE(changed);
        REQUIRE(tree.size() == 1);
        CHECK(tree[0].level == 0);
        REQUIRE_NOTHROW(tree.verify());
    }

    SECTION("Graded Coarsening (Misaligned Families)") {
        // Test robustness against misaligned families in the linear array.
        TestType tree(5);

        // Create Graded Mesh: Refine all to L1, then refine the LAST L1 node to L2.
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
        }))
            ;

        // Expectation:
        // L1 Nodes: (2^DIM - 1)
        // L2 Nodes: (2^DIM) (Refined from the last L1)
        size_t expected_size = ((1ULL << DIM) - 1) + (1ULL << DIM);
        REQUIRE(tree.size() == expected_size);
        REQUIRE_NOTHROW(tree.verify());

        // Force coarsening.
        // The L2 family should merge. The misaligned L1 nodes should eventually merge.
        int passes = 0;
        while (tree.coarsen([&](const Node&, int) { return false; })) {
            passes++;
            tree.verify();
        }

        // Should return to root
        REQUIRE(tree.size() == 1);
        CHECK(tree[0].level == 0);
    }
}

TEMPLATE_TEST_CASE("2:1 Balance & Ripple Algorithm (Holke §3.3)", "[balance]", Quadtree, Octree) {
    constexpr int DIM = (std::is_same<TestType, Quadtree>::value) ? 2 : 3;

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

        while (tree.refine(center_oracle))
            ;

        check_balance_properties(tree);
    }

    SECTION("Random Cloud Stress Test") {
        // Sparse pseudo-random refinement (a multiplicative hash of the code) so
        // neighbouring leaves end up several levels apart before balancing.
        TestType tree(8);
        auto cloud_oracle = [&](const Node& n, int) {
            if (n.level >= 6)
                return false;
            return ((n.code.value * 0x9e3779b97f4a7c15ULL) >> 40) % 4 == 0;
        };

        for (int i = 0; i < 8; ++i)
            tree.refine(cloud_oracle);

        check_balance_properties(tree);
    }
}

TEST_CASE("Full 2:1 balance resolves 3D edge-diagonal jumps", "[balance][edge]") {
    // Regression for the AMR<->DIC bridge (code/+mesh/importFromAmr.m). Face-only
    // balance leaves an edge-diagonal 2-level jump -- a coarse cell sharing only an
    // EDGE with cells two levels finer -- which drops a node at the coarse edge's
    // quarter point, inexpressible as a two-parent midpoint constraint. Refine the
    // octant [4,8]x[4,8]x[0,4] of an [0,8]^3 domain to level 3; the level-1 corner
    // cell [0,4]^3 then sits edge-diagonal to level-3 cells with no level-3 FACE
    // neighbour, so face-only balance leaves it untouched. Full balance must not.
    Octree tree(3);
    auto oracle = [&](const Node& n, int) {
        if (n.level >= 3)
            return false;
        auto c = tree.decode(n.code);
        uint64_t size = 1ULL << (3 - n.level);
        bool ix = c[0].value < 8 && c[0].value + size > 4;
        bool iy = c[1].value < 8 && c[1].value + size > 4;
        bool iz = c[2].value < 4;  // z in [0,4)
        return ix && iy && iz;
    };
    while (tree.refine(oracle))
        ;

    // The fixture genuinely contains the edge violation (guards against a vacuous
    // test if the oracle geometry ever drifts): face-only balance would leave it.
    REQUIRE(count_balance_violations(tree) > 0);
    tree.balance();
    tree.verify();
    REQUIRE(count_balance_violations(tree) == 0);
}

TEMPLATE_TEST_CASE("Active-front balance parity (byte-identical vs balance_ref)",
                   "[balance][parity]",
                   Quadtree,
                   Octree) {
    constexpr int DIM = (std::is_same<TestType, Quadtree>::value) ? 2 : 3;

    // Tree-independent oracles (decode the Node's own code) so two trees refine identically.
    auto decode = [](MortonCode code) {
        if constexpr (DIM == 2)
            return morton::decode_2d(code);
        else
            return morton::decode_3d(code);
    };

    auto run_parity = [&](const char* name, int max_lvl, auto oracle, int steps) {
        TestType ref(max_lvl), act(max_lvl);
        for (int i = 0; i < steps; ++i) {
            ref.refine(oracle);
            act.refine(oracle);
        }

        ref.balance_ref();  // whole-mesh baseline
        act.balance();      // active-front

        INFO("fixture=" << name << " DIM=" << DIM);
        REQUIRE(ref.size() == act.size());
        for (size_t i = 0; i < ref.size(); ++i) {
            REQUIRE(ref[i].code.value == act[i].code.value);
            REQUIRE(ref[i].level == act[i].level);
        }
        REQUIRE(count_balance_violations(act) == 0);  // independent geometric check
        REQUIRE(ref.last_balance_iters == act.last_balance_iters);
    };

    SECTION("Tower (deep ripple)") {
        int L = 6;
        run_parity(
            "tower",
            L,
            [L, decode](const Node& n, int) {
                uint64_t mid = (1ULL << L) / 2, size = 1ULL << (L - n.level);
                auto c = decode(n.code);
                bool hit = true;
                for (int k = 0; k < DIM; ++k)
                    if (!(c[k].value <= mid && c[k].value + size > mid))
                        hit = false;
                return hit && n.level < L;
            },
            L);
    }
    SECTION("Random cloud") {
        run_parity(
            "cloud",
            8,
            [](const Node& n, int) {
                if (n.level >= 5)
                    return false;
                return (n.code.value % 7 == 0 || n.code.value % 13 == 0);
            },
            8);
    }
    SECTION("Deterministic hash") {
        run_parity(
            "random",
            5,
            [](const Node& n, int) {
                if (n.level >= 4)
                    return false;
                return (n.code.value * 0x9e3779b97f4a7c15ULL) % 3 == 0;
            },
            5);
    }
}

TEST_CASE("Safety & Edge Cases", "[safety]") {
    // Only need one dimension type to test general logic logic
    using TestType = Quadtree;

    SECTION("Empty Tree Violation") {
        // This is tricky to test since the constructor guarantees a root node.
        // We would need to manually clear the private vectors, which we can't do.
        // Instead, we verify the constructor post-condition.
        TestType tree(5);
        REQUIRE(tree.size() > 0);
        REQUIRE_NOTHROW(tree.verify());
    }

    SECTION("Max Level Constraints") {
        // 2D Max is 31
        REQUIRE_NOTHROW(Quadtree(31));
        // 3D Max is 21
        REQUIRE_NOTHROW(Octree(21));
    }
}
