/**
 * @file test_oracle.cpp
 * @brief ScalarFieldOracle's Dörfler marking and coarsening.
 */

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "physics.hpp"
#include "test_util.hpp"

using namespace amr;
using namespace amr::test;

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
