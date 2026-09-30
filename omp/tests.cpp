/**
 * @file tests.cpp
 * @brief Validation Suite for Linear AMR System (Catch2 v3)
 * * @details
 * Validates the core invariants described in AMR literature:
 * 1. **Bijection**: Morton coding must be lossless and reversible.
 * 2. **Partition of Unity**: The sum of leaf volumes must exactly equal the domain volume.
 * 3. **2:1 Balance**: Verified using an independent geometric search (brute-force) to
 * confirm the efficiency of the optimized Ripple algorithm.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "../common/mesh_io.hpp"
#include "../common/viz.hpp"
#include "core.hpp"
#include "physics.hpp"
#include "tree.hpp"

using namespace amr;
using namespace Catch::Matchers;

// ==================================================================================
// HELPER: Robust Balance Verification (Public API Version)
// ==================================================================================

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

// ==================================================================================
// TEST SUITE
// ==================================================================================

TEST_CASE("Space-Filling Curve Bijection (Holke §3.1)", "[core][sfc]") {
    // Deterministic Random Generator
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint32_t> dist(0, std::numeric_limits<uint32_t>::max());

    SECTION("2D Morton Encoding (Strong Types)") {
        for (int i = 0; i < 10000; ++i) {
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
        for (int i = 0; i < 10000; ++i) {
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

TEMPLATE_TEST_CASE("Linear Tree Invariants (Burstedde §2.2)", "[tree]", Quadtree, Octree) {
    constexpr int DIM = (std::is_same<TestType, Quadtree>::value) ? 2 : 3;

    SECTION("Completeness & Partition of Unity") {
        int max_lvl = 5;
        TestType tree(max_lvl);
        std::mt19937 rng(123);

        // Create a non-uniform random mesh
        tree.refine([&](const Node& n, int) { return (n.level < 4 && (rng() % 3 == 0)); });

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

TEST_CASE("get_neighbor_code returns the sentinel past the +domain boundary (core)",
          "[core][neighbor][boundary]") {
    // 3D at the maximum supported level: a coarse boundary cell's +y/+z neighbour
    // used to wrap to the origin (the carry ran off the top of the 64-bit word),
    // and +x used to land above every valid code, instead of the OOB sentinel.
    Octree tree(21);
    const uint64_t maxc = 1ULL << 21;
    const uint64_t step2 = 1ULL << (21 - 2);  // a level-2 cell's span per axis
    const uint64_t hi = maxc - step2;         // max aligned level-2 coordinate

    auto at = [&](uint64_t x, uint64_t y, uint64_t z) {
        Octree::Point p{Coordinate{static_cast<uint32_t>(x)},
                        Coordinate{static_cast<uint32_t>(y)},
                        Coordinate{static_cast<uint32_t>(z)}};
        return tree.encode(p);
    };

    SECTION("+x/+y/+z at the far corner are all out of bounds") {
        MortonCode corner = at(hi, hi, hi);
        REQUIRE(tree.get_neighbor_code(corner, 2, {1, 0, 0}).value == UINT64_MAX);
        REQUIRE(tree.get_neighbor_code(corner, 2, {0, 1, 0}).value == UINT64_MAX);
        REQUIRE(tree.get_neighbor_code(corner, 2, {0, 0, 1}).value == UINT64_MAX);
    }
    SECTION("interior +neighbours still resolve; -dir boundary is out of bounds") {
        MortonCode interior = at(step2, step2, step2);
        REQUIRE(tree.get_neighbor_code(interior, 2, {1, 0, 0}).value != UINT64_MAX);
        MortonCode origin = at(0, 0, 0);
        REQUIRE(tree.get_neighbor_code(origin, 2, {-1, 0, 0}).value == UINT64_MAX);
    }
    SECTION("a level-0 cell has no neighbour (and no 1<<64 shift UB)") {
        REQUIRE(tree.get_neighbor_code(MortonCode{0}, 0, {1, 0, 0}).value == UINT64_MAX);
        REQUIRE(tree.get_neighbor_code(MortonCode{0}, 0, {0, 1, 0}).value == UINT64_MAX);
    }
}

namespace {
/// morton::neighbor_code's oracle: decode, move one cell along each axis,
/// check the domain, encode.
template <int DIM>
uint64_t neighbor_by_coordinates(uint64_t code, int level, int max_level, const int* dir) {
    std::array<Coordinate, DIM> x;
    if constexpr (DIM == 2)
        x = morton::decode_2d(MortonCode{code});
    else
        x = morton::decode_3d(MortonCode{code});
    const int64_t side = int64_t{1} << (max_level - level);
    const int64_t extent = int64_t{1} << max_level;
    std::array<Coordinate, DIM> y;
    for (int k = 0; k < DIM; ++k) {
        const int64_t v = int64_t{x[static_cast<std::size_t>(k)].value} + dir[k] * side;
        if (v < 0 || v >= extent)
            return UINT64_MAX;
        y[static_cast<std::size_t>(k)] = Coordinate{static_cast<uint32_t>(v)};
    }
    if constexpr (DIM == 2)
        return morton::encode_2d(y[0], y[1]).value;
    else
        return morton::encode_3d(y[0], y[1], y[2]).value;
}

template <int DIM>
void check_neighbor(uint64_t code, int level, int max_level, const int* dir) {
    const uint64_t got = morton::neighbor_code<DIM>(code, level, max_level, dir);
    const uint64_t want = neighbor_by_coordinates<DIM>(code, level, max_level, dir);
    if (got != want) {
        INFO("code " << code << " level " << level << " max_level " << max_level << " dir "
                     << dir[0] << "," << dir[1] << "," << (DIM == 3 ? dir[2] : 0));
        CHECK(got == want);
    }
}
}  // namespace

TEMPLATE_TEST_CASE_SIG("neighbor_code matches decode, move, encode (core)",
                       "[core][neighbor]",
                       ((int DIM), DIM),
                       2,
                       3) {
    std::vector<std::array<int, 3>> dirs;
    for (int a = -1; a <= 1; ++a)
        for (int b = -1; b <= 1; ++b)
            for (int c = (DIM == 3 ? -1 : 0); c <= (DIM == 3 ? 1 : 0); ++c)
                dirs.push_back({a, b, c});

    SECTION("every aligned cell of every level of a small domain, every direction") {
        const int max_level = DIM == 2 ? 5 : 3;
        for (int level = 0; level <= max_level; ++level) {
            const uint32_t side = 1u << (max_level - level);
            const uint32_t extent = 1u << max_level;
            for (uint32_t i = 0; i < extent; i += side)
                for (uint32_t j = 0; j < extent; j += side)
                    for (uint32_t k = 0; k < (DIM == 3 ? extent : 1u); k += side) {
                        const uint64_t code =
                            DIM == 2
                                ? morton::encode_2d(Coordinate{i}, Coordinate{j}).value
                                : morton::encode_3d(Coordinate{i}, Coordinate{j}, Coordinate{k})
                                      .value;
                        for (const auto& d : dirs)
                            check_neighbor<DIM>(code, level, max_level, d.data());
                    }
        }
    }
    SECTION("random cells of the deepest trees: 62 bits in 2D, 63 in 3D") {
        const int max_level = morton::max_level_limit(DIM);
        std::mt19937_64 rng(DIM);
        for (int n = 0; n < 20000; ++n) {
            const int level = static_cast<int>(rng() % static_cast<uint64_t>(max_level + 1));
            const uint64_t align = ~((uint64_t{1} << (max_level - level)) - 1);
            const uint64_t limit = (uint64_t{1} << max_level) - 1;
            std::array<Coordinate, 3> c{};
            for (auto& v : c)
                v = Coordinate{static_cast<uint32_t>(rng() & limit & align)};
            const uint64_t code = DIM == 2 ? morton::encode_2d(c[0], c[1]).value
                                           : morton::encode_3d(c[0], c[1], c[2]).value;
            check_neighbor<DIM>(code, level, max_level, dirs[rng() % dirs.size()].data());
        }
    }
}

TEMPLATE_TEST_CASE_SIG("neighbor_code rejects what it cannot answer (core)",
                       "[core][neighbor]",
                       ((int DIM), DIM),
                       2,
                       3) {
    // No cell's code is UINT64_MAX, so the sentinel cannot be confused with a
    // neighbour: every code of the deepest tree stays below 2^63.
    const int deepest = morton::max_level_limit(DIM);
    const uint32_t far = (1u << deepest) - 1;
    const uint64_t corner =
        DIM == 2 ? morton::encode_2d(Coordinate{far}, Coordinate{far}).value
                 : morton::encode_3d(Coordinate{far}, Coordinate{far}, Coordinate{far}).value;
    CHECK(corner < (uint64_t{1} << 63));
    const int west[3] = {-1, 0, 0};
    CHECK(morton::neighbor_code<DIM>(corner, deepest, deepest, west) ==
          neighbor_by_coordinates<DIM>(corner, deepest, deepest, west));
    CHECK(morton::neighbor_code<DIM>(corner, deepest, deepest, west) != UINT64_MAX);
    // Out-of-range levels and trees deeper than the limit get the sentinel.
    const int east[3] = {1, 0, 0};
    CHECK(morton::neighbor_code<DIM>(0, 4, 3, east) == UINT64_MAX);
    CHECK(morton::neighbor_code<DIM>(0, -1, 3, east) == UINT64_MAX);
    CHECK(morton::neighbor_code<DIM>(0, 1, deepest + 1, east) == UINT64_MAX);
    CHECK_THROWS_AS(LinearTree<DIM>(deepest + 1), std::invalid_argument);
    CHECK_THROWS_AS(LinearTree<DIM>(-1), std::invalid_argument);
    CHECK_NOTHROW(LinearTree<DIM>(deepest));
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

TEMPLATE_TEST_CASE("Binary mesh+field format round-trip (mesh_io C.2)",
                   "[mesh_io][integration]",
                   Quadtree,
                   Octree) {
    constexpr int DIM = (std::is_same<TestType, Quadtree>::value) ? 2 : 3;

    // Build a non-uniform mesh.
    int max_lvl = 6;
    TestType tree(max_lvl);
    std::mt19937_64 rng(7);
    for (int i = 0; i < 4; ++i)
        tree.refine([&](const Node& n, int) { return n.level < 5 && (rng() % 3 == 0); });

    // Pack into a MeshData with a non-trivial bbox + two fields (f64 and f32).
    mesh_io::MeshData m;
    m.dim = DIM;
    m.max_level = static_cast<uint32_t>(max_lvl);
    m.origin = {{1.5, -2.0, 3.25}};
    m.size = {{100.0, 50.0, 12.5}};
    for (const auto& node : tree) {
        m.codes.push_back(node.code.value);
        m.levels.push_back(static_cast<uint8_t>(node.level));
    }
    mesh_io::Field err{"dic_error", true, {}};
    mesh_io::Field lvl{"level_f32", false, {}};
    for (size_t i = 0; i < m.codes.size(); ++i) {
        err.values.push_back(std::sin(static_cast<double>(i)) * 1.0e-3);
        lvl.values.push_back(static_cast<double>(m.levels[i]));
    }
    m.fields = {err, lvl};

    auto path =
        (std::filesystem::temp_directory_path() / ("amr_mesh_io_" + std::to_string(DIM) + "d.bin"))
            .string();

    SECTION("write/read preserves geometry, codes, levels, and fields") {
        mesh_io::write(path, m);
        mesh_io::MeshData r = mesh_io::read(path);

        REQUIRE(r.dim == m.dim);
        REQUIRE(r.max_level == m.max_level);
        for (uint32_t k = 0; k < m.dim; ++k) {
            REQUIRE(r.origin[k] == m.origin[k]);  // f64 bbox is exact
            REQUIRE(r.size[k] == m.size[k]);
        }
        REQUIRE(r.codes == m.codes);  // bit-exact
        REQUIRE(r.levels == m.levels);
        REQUIRE(r.elem_type == mesh_io::default_elem_type(DIM));  // unset -> dim default

        REQUIRE(r.fields.size() == 2);
        REQUIRE(r.fields[0].name == "dic_error");
        REQUIRE(r.fields[0].f64);
        REQUIRE(r.fields[0].values == err.values);  // f64 exact
        REQUIRE(r.fields[1].name == "level_f32");
        REQUIRE_FALSE(r.fields[1].f64);
        for (size_t i = 0; i < lvl.values.size(); ++i)
            REQUIRE_THAT(r.fields[1].values[i], WithinAbs(lvl.values[i], 1e-5));  // f32 round-off
        std::filesystem::remove(path);
    }

    SECTION("rejects a corrupt magic") {
        {
            std::ofstream bad(path, std::ios::binary);
            bad << "XXXXnonsense";
        }
        REQUIRE_THROWS_AS(mesh_io::read(path), std::runtime_error);
        std::filesystem::remove(path);
    }
}

TEST_CASE("Binary mesh+field format edge cases (mesh_io C.2)", "[mesh_io]") {
    auto path = (std::filesystem::temp_directory_path() / "amr_mesh_io_edge.bin").string();

    SECTION("empty mesh and field-less round-trip") {
        mesh_io::MeshData m;
        m.dim = 3;
        m.max_level = 10;
        mesh_io::write(path, m);
        mesh_io::MeshData r = mesh_io::read(path);
        REQUIRE(r.codes.empty());
        REQUIRE(r.fields.empty());
        REQUIRE(r.dim == 3);
        std::filesystem::remove(path);
    }

    SECTION("padding: n not a multiple of 8 round-trips with a trailing field") {
        mesh_io::MeshData m;
        m.dim = 2;
        m.max_level = 4;
        m.codes = {0, 1, 2, 3, 4};  // n = 5 -> 3 pad bytes before n_fields
        m.levels = {1, 1, 1, 1, 1};
        m.fields = {mesh_io::Field{"f", true, {0.1, 0.2, 0.3, 0.4, 0.5}}};
        mesh_io::write(path, m);
        mesh_io::MeshData r = mesh_io::read(path);
        REQUIRE(r.codes == m.codes);
        REQUIRE(r.fields.size() == 1);
        REQUIRE(r.fields[0].values == m.fields[0].values);
        std::filesystem::remove(path);
    }
}

TEST_CASE("AMR1 element-type tag (mesh_io #12)", "[mesh_io][elem_type]") {
    namespace mio = amr::mesh_io;
    auto path = (std::filesystem::temp_directory_path() / "amr_mesh_io_etype.bin").string();

    SECTION("unset tag defaults to the dim simplex on write") {
        for (uint32_t dim : {2u, 3u}) {
            mio::MeshData m;
            m.dim = dim;
            mio::write(path, m);
            mio::MeshData r = mio::read(path);
            REQUIRE(r.elem_type == mio::default_elem_type(dim));
        }
        std::filesystem::remove(path);
    }

    SECTION("an explicit tag round-trips") {
        mio::MeshData m;
        m.dim = 2;
        m.elem_type = mio::ElemType::Q4;  // overriding the T3 default
        mio::write(path, m);
        mio::MeshData r = mio::read(path);
        REQUIRE(r.elem_type == mio::ElemType::Q4);
        std::filesystem::remove(path);
    }

    SECTION("a version-1 file (no tag) reads back as the dim default") {
        // Hand-write a minimal v1 header (empty mesh, no fields).
        {
            std::ofstream os(path, std::ios::binary | std::ios::trunc);
            os.write("AMR1", 4);
            mio::detail::put<uint32_t>(os, 1u);  // version 1: no elem_type on disk
            mio::detail::put<uint32_t>(os, 3u);  // dim
            mio::detail::put<uint32_t>(os, 5u);  // max_level
            mio::detail::put<uint64_t>(os, 0u);  // n_leaves
            for (int k = 0; k < 3; ++k)
                mio::detail::put<double>(os, 0.0);  // origin
            for (int k = 0; k < 3; ++k)
                mio::detail::put<double>(os, 1.0);  // size
            mio::detail::put<uint32_t>(os, 0u);     // n_fields
        }
        mio::MeshData r = mio::read(path);
        REQUIRE(r.elem_type == mio::ElemType::T4);  // dim==3 default
        std::filesystem::remove(path);
    }

    SECTION("an unknown element code is rejected") {
        {
            std::ofstream os(path, std::ios::binary | std::ios::trunc);
            os.write("AMR1", 4);
            mio::detail::put<uint32_t>(os, 2u);   // version 2
            mio::detail::put<uint32_t>(os, 2u);   // dim
            mio::detail::put<uint32_t>(os, 1u);   // max_level
            mio::detail::put<uint32_t>(os, 99u);  // bogus elem_type
            mio::detail::put<uint64_t>(os, 0u);   // n_leaves
            for (int k = 0; k < 2; ++k)
                mio::detail::put<double>(os, 0.0);
            for (int k = 0; k < 2; ++k)
                mio::detail::put<double>(os, 1.0);
            mio::detail::put<uint32_t>(os, 0u);
        }
        REQUIRE_THROWS_AS(mio::read(path), std::runtime_error);
        std::filesystem::remove(path);
    }
}

TEST_CASE("AMR1 reader rejects malformed input cleanly (mesh_io)", "[mesh_io][robustness]") {
    namespace mio = amr::mesh_io;
    auto path = (std::filesystem::temp_directory_path() / "amr_mesh_io_bad.bin").string();

    SECTION("an unknown field dtype is rejected, not silently read as float32") {
        {
            std::ofstream os(path, std::ios::binary | std::ios::trunc);
            os.write("AMR1", 4);
            mio::detail::put<uint32_t>(os, 2u);  // version 2
            mio::detail::put<uint32_t>(os, 2u);  // dim
            mio::detail::put<uint32_t>(os, 1u);  // max_level
            mio::detail::put<uint32_t>(os, 0u);  // elem_type T3
            mio::detail::put<uint64_t>(os, 0u);  // n_leaves = 0
            for (int k = 0; k < 2; ++k)
                mio::detail::put<double>(os, 0.0);
            for (int k = 0; k < 2; ++k)
                mio::detail::put<double>(os, 1.0);
            mio::detail::put<uint32_t>(os, 1u);  // n_fields = 1
            mio::detail::put<uint32_t>(os, 1u);  // name_len = 1
            os.put('f');                         // name
            mio::detail::put<uint32_t>(os, 2u);  // bogus dtype (only 0/1 valid)
        }
        REQUIRE_THROWS_AS(mio::read(path), std::runtime_error);
        std::filesystem::remove(path);
    }

    SECTION("a hostile leaf count throws runtime_error, not bad_alloc") {
        {
            std::ofstream os(path, std::ios::binary | std::ios::trunc);
            os.write("AMR1", 4);
            mio::detail::put<uint32_t>(os, 2u);     // version 2
            mio::detail::put<uint32_t>(os, 2u);     // dim
            mio::detail::put<uint32_t>(os, 1u);     // max_level
            mio::detail::put<uint32_t>(os, 0u);     // elem_type
            mio::detail::put<uint64_t>(os, ~0ull);  // n_leaves = 2^64-1 (hostile)
            for (int k = 0; k < 2; ++k)
                mio::detail::put<double>(os, 0.0);
            for (int k = 0; k < 2; ++k)
                mio::detail::put<double>(os, 1.0);
        }
        REQUIRE_THROWS_AS(mio::read(path), std::runtime_error);  // bounded before allocation
        std::filesystem::remove(path);
    }
}

// Snapshot the current leaf set in Morton (iteration) order -- the order the DIC
// error array is expected to align with.
template <typename Tree>
static void snapshot(const Tree& tree, std::vector<uint64_t>& codes, std::vector<uint8_t>& levels) {
    codes.clear();
    levels.clear();
    for (const auto& n : tree) {
        codes.push_back(n.code.value);
        levels.push_back(static_cast<uint8_t>(n.level));
    }
}

TEST_CASE("ScalarFieldOracle Dörfler marking + coarsening (oracle #11)", "[oracle][dorfler]") {
    // A quadtree refined once: 4 level-1 leaves at Morton codes 0,64,128,192
    // (max_level 4 => a level-1 cell spans 1<<(2*(4-1)) = 64 in code value).
    const int max_lvl = 4;
    Quadtree tree(max_lvl);
    tree.refine([](const Node&, int) { return true; });  // one pass: root -> 4 leaves
    REQUIRE(tree.size() == 4);

    std::vector<uint64_t> codes;
    std::vector<uint8_t> levels;
    snapshot(tree, codes, levels);
    REQUIRE(codes == std::vector<uint64_t>{0, 64, 128, 192});

    SECTION("Dörfler marks the smallest high-error set covering theta") {
        // error = {10,1,1,1}: total sq = 103, half = 51.5. The single 10 (sq 100)
        // already covers > half, so only leaf 0 is marked at theta = 0.5.
        std::vector<double> err{10, 1, 1, 1};
        ScalarFieldOracle<2> oracle(codes, levels, err, 0.5, 0.0, 0, max_lvl);
        int marked = 0;
        for (const auto& n : tree)
            if (oracle(n, max_lvl)) {
                ++marked;
                REQUIRE(n.code.value == 0);  // only the high-error leaf
            }
        REQUIRE(marked == 1);
    }

    SECTION("theta = 1 marks every leaf; all-zero error marks none") {
        std::vector<double> err{10, 1, 1, 1};
        ScalarFieldOracle<2> all(codes, levels, err, 1.0, 0.0, 0, max_lvl);
        int n_all = 0;
        for (const auto& n : tree)
            n_all += all(n, max_lvl);
        REQUIRE(n_all == 4);

        std::vector<double> zero(4, 0.0);
        ScalarFieldOracle<2> none(codes, levels, zero, 0.5, 0.0, 0, max_lvl);
        int n_none = 0;
        for (const auto& n : tree)
            n_none += none(n, max_lvl);
        REQUIRE(n_none == 0);
    }

    SECTION("refine splits exactly the marked leaves, one level, and terminates") {
        std::vector<double> err{10, 1, 1, 1};
        ScalarFieldOracle<2> oracle(codes, levels, err, 0.5, 0.0, 0, max_lvl);
        while (tree.refine(oracle))  // safe under while(): must not refine without bound
            ;
        tree.verify();
        // leaf 0 (1) split into 4; leaves 1..3 (3) untouched => 3 + 4 = 7.
        REQUIRE(tree.size() == 7);
    }

    SECTION("the base coarse_level is forced everywhere regardless of error") {
        std::vector<double> zero(4, 0.0);
        ScalarFieldOracle<2> oracle(codes, levels, zero, 0.5, 0.0, /*coarse*/ 3, max_lvl);
        while (tree.refine(oracle))
            ;
        tree.verify();
        for (const auto& n : tree)
            REQUIRE(n.level >= 3);  // driven to the base level
    }

    SECTION("coarsen merges a complete low-error family, keeping high-error siblings") {
        // Refine only leaf 0 into a level-2 family; give it low error, the rest high.
        tree.refine([](const Node& n, int) { return n.code.value == 0 && n.level == 1; });
        REQUIRE(tree.size() == 7);
        snapshot(tree, codes, levels);
        // Morton order: 0,16,32,48 (the level-2 family) then 64,128,192 (level 1).
        std::vector<double> err{1, 1, 1, 1, 10, 10, 10};
        // theta_coarsen = 0.01: total sq = 4 + 300 = 304, 1% = 3.04; the bottom tail
        // stops at the 4th unit-error leaf => coarsen cutoff = 1, so the family (max
        // child error 1) is coarsenable but the 10s are not.
        ScalarFieldOracle<2> oracle(codes, levels, err, 0.5, 0.01, 0, max_lvl);
        while (tree.coarsen(oracle.coarsener()))
            ;
        tree.verify();
        REQUIRE(tree.size() == 4);  // family merged back; the three 10-leaves remain
    }

    SECTION("constructor rejects malformed input") {
        std::vector<double> err{1, 1, 1, 1};
        REQUIRE_THROWS_AS(ScalarFieldOracle<2>(codes, levels, {1, 1, 1}, 0.5, 0.0, 0, max_lvl),
                          std::invalid_argument);  // size mismatch
        std::vector<uint64_t> unsorted{192, 0, 64, 128};
        REQUIRE_THROWS_AS(ScalarFieldOracle<2>(unsorted, levels, err, 0.5, 0.0, 0, max_lvl),
                          std::invalid_argument);  // not Morton-sorted
        REQUIRE_THROWS_AS(ScalarFieldOracle<2>(codes, levels, err, 1.5, 0.0, 0, max_lvl),
                          std::invalid_argument);  // theta out of range
        REQUIRE_THROWS_AS(ScalarFieldOracle<2>(codes, levels, {-1, 1, 1, 1}, 0.5, 0.0, 0, max_lvl),
                          std::invalid_argument);  // negative error (DIC error is >= 0)
        REQUIRE_THROWS_AS(
            ScalarFieldOracle<2>(codes, levels, err, 0.5, 0.0, /*coarse*/ 4, /*fine*/ 2),
            std::invalid_argument);  // coarse_level > fine_level
    }
}

TEST_CASE("ScalarFieldOracle marks a minimum-size set covering theta", "[oracle][dorfler]") {
    // Dorfler bulk marking [Doerfler1996] in its squared form: the marked set
    // covers theta of the total squared error and no smaller set does. Checked
    // against every subset of 16 leaves (2^16) for distinct random errors.
    const int max_lvl = 4;
    Quadtree tree(max_lvl);
    tree.refine([](const Node&, int) { return true; });
    tree.refine([](const Node&, int) { return true; });
    std::vector<uint64_t> codes;
    std::vector<uint8_t> levels;
    snapshot(tree, codes, levels);
    REQUIRE(codes.size() == 16);

    std::mt19937 rng(20260929);
    std::uniform_real_distribution<double> dist(0.1, 10.0);
    std::vector<double> err(16);
    for (auto& e : err)
        e = dist(rng);
    double total = 0.0;
    for (double e : err)
        total += e * e;

    for (double theta : {0.1, 0.3, 0.5, 0.7, 0.9}) {
        INFO("theta=" << theta);
        ScalarFieldOracle<2> oracle(codes, levels, err, theta, 0.0, 0, max_lvl);
        double covered = 0.0;
        int marked = 0;
        size_t i = 0;
        for (const auto& n : tree) {
            if (oracle(n, max_lvl)) {
                covered += err[i] * err[i];
                ++marked;
            }
            ++i;
        }
        REQUIRE(covered >= theta * total);

        int smallest = 17;
        for (uint32_t mask = 1; mask < (1u << 16); ++mask) {
            double s = 0.0;
            for (int k = 0; k < 16; ++k)
                if (mask & (1u << k))
                    s += err[k] * err[k];
            if (s >= theta * total)
                smallest = std::min(smallest, __builtin_popcount(mask));
        }
        REQUIRE(marked == smallest);
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

TEST_CASE("viz::write_vtk emits UNSTRUCTURED_GRID cells, not a point cloud", "[viz]") {
    // Guards the single-source writer's core contract: one VTK cell per leaf with
    // the right cell type. The per-backend copies this replaced had diverged to a
    // VTK_VERTEX point cloud (one point per leaf) and a mislabelled VTK_VOXEL --
    // both silently wrong in ParaView. Build a LeafView by hand (no tree needed)
    // and read the file back.
    namespace fs = std::filesystem;
    auto slurp = [](const std::string& p) {
        std::ifstream f(p);
        std::stringstream ss;
        ss << f.rdbuf();
        return ss.str();
    };

    SECTION("2D -> VTK_QUAD") {
        std::vector<uint64_t> codes = {0, 1, 2, 3};  // four level-1 quads
        std::vector<uint8_t> levels = {1, 1, 1, 1};
        viz::LeafView v{codes.data(), levels.data(), 4, 1, 2};
        const std::string path = (fs::temp_directory_path() / "amr_viz_2d.vtk").string();
        viz::write_vtk(path, v);
        const std::string s = slurp(path);
        fs::remove(path);
        REQUIRE(s.find("DATASET UNSTRUCTURED_GRID") != std::string::npos);
        REQUIRE(s.find("POINTS 16 double") != std::string::npos);  // 4 quads x 4 corners
        REQUIRE(s.find("CELLS 4 20") != std::string::npos);        // n, n*(4+1)
        REQUIRE(s.find("CELL_TYPES 4\n9") != std::string::npos);   // 9 = VTK_QUAD
    }

    SECTION("3D -> VTK_HEXAHEDRON") {
        std::vector<uint64_t> codes = {0, 1, 2, 3, 4, 5, 6, 7};  // eight level-1 hexes
        std::vector<uint8_t> levels = {1, 1, 1, 1, 1, 1, 1, 1};
        viz::LeafView v{codes.data(), levels.data(), 8, 1, 3};
        const std::string path = (fs::temp_directory_path() / "amr_viz_3d.vtk").string();
        viz::write_vtk(path, v);
        const std::string s = slurp(path);
        fs::remove(path);
        REQUIRE(s.find("DATASET UNSTRUCTURED_GRID") != std::string::npos);
        REQUIRE(s.find("POINTS 64 double") != std::string::npos);  // 8 hexes x 8 corners
        REQUIRE(s.find("CELLS 8 72") != std::string::npos);        // n, n*(8+1)
        REQUIRE(s.find("CELL_TYPES 8\n12") != std::string::npos);  // 12 = VTK_HEXAHEDRON
    }
}
