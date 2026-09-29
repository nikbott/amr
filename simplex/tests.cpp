/**
 * @file tests.cpp
 * @brief Catch2 suite for the simplicial refinement module (simplex.hpp).
 *
 * @details Checks refinement and balance against independent oracles:
 * children partition the parent's measure in equal parts with its
 * orientation [Bey1995]; the octahedron diagonal is the best of the three,
 * enumerated here without the module's pattern tables [Zhang1995]; new nodes
 * sit bit-exactly at their parents' midpoint; and, by brute force over every
 * element edge, the hanging-node list is exactly the set of nodes sitting at
 * the midpoint of an edge some element still has. The balance closure is
 * compared with a naive fixed-point transcription of the MATLAB loop.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "simplex.hpp"
#include "test_util.hpp"

using namespace amr::simplex;
using namespace amr::simplex::test;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

TEST_CASE("A regular tetrahedron has mean-ratio quality 1 at any scale", "[simplex]") {
    const double s = GENERATE(1e-3, 1.0, 555.0);
    const Point a{s, s, s}, b{s, -s, -s}, c{-s, s, -s}, d{-s, -s, s};
    CHECK_THAT(detail::tet_quality(a, b, c, d), WithinRel(1.0, 1e-12));
}

TEST_CASE("Children split the parent's measure equally and keep its orientation", "[simplex]") {
    const int dim = GENERATE(2, 3);
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (int trial = 0; trial < 200; ++trial) {
        Mesh m;
        m.dim = dim;
        for (int i = 0; i <= dim; ++i)
            m.pos.push_back({u(rng), u(rng), dim == 3 ? u(rng) : 0.0});
        std::vector<Index> v(static_cast<std::size_t>(dim + 1));
        std::iota(v.begin(), v.end(), 0);
        add_element(m, v);
        const double parent = signed_measure(m, 0);
        if (parent < 1e-3)
            continue;  // near-degenerate draw

        const std::vector<Index> list{0};
        const auto r = refine(m, list);
        REQUIRE(r.mesh.num_elements() == (dim == 2 ? 4 : 8));
        REQUIRE(r.mesh.pos.size() == m.pos.size() + (dim == 2 ? 3 : 6));
        for (Index e = 0; e < r.mesh.num_elements(); ++e)
            CHECK_THAT(signed_measure(r.mesh, e), WithinRel(parent / (dim == 2 ? 4 : 8), 1e-9));
        CHECK(r.mesh.hn.empty());  // no neighbour keeps a refined edge
    }
}

TEST_CASE("The octahedron is cut along the diagonal with the best worst child", "[simplex]") {
    std::mt19937 rng(11);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (int trial = 0; trial < 500; ++trial) {
        Mesh m;
        m.dim = 3;
        std::array<Point, 4> c;
        for (auto& p : c) {
            p = {u(rng), u(rng), u(rng)};
            m.pos.push_back(p);
        }
        add_element(m, {0, 1, 2, 3});
        if (signed_measure(m, 0) < 1e-3)
            continue;

        const std::vector<Index> list{0};
        const auto r = refine(m, list);
        double worst_inner = 1e300;
        for (Index e = 1; e < 5; ++e) {  // children 1..4 fill the octahedron
            const auto v = r.mesh.element(e);
            const auto p = [&](std::size_t i) {
                return r.mesh.pos[static_cast<std::size_t>(v[i])];
            };
            worst_inner = std::min(worst_inner, detail::tet_quality(p(0), p(1), p(2), p(3)));
        }
        CHECK_THAT(worst_inner, WithinRel(best_octahedron_worst_quality(c), 1e-12));
    }
}

TEST_CASE("A shared edge's midpoint hangs until both sides are refined", "[simplex]") {
    const int dim = GENERATE(2, 3);
    Mesh m;
    m.dim = dim;
    if (dim == 2) {
        m.pos = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}};
        add_element(m, {0, 1, 2});
        add_element(m, {1, 3, 2});
    } else {
        m.pos = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 1}};
        add_element(m, {0, 1, 2, 3});
        add_element(m, {1, 2, 3, 4});
    }
    const std::size_t shared_edges = dim == 2 ? 1 : 3;

    const std::vector<Index> first{0};
    const auto once = refine(m, first);
    REQUIRE(once.mesh.hn.size() == shared_edges);
    for (const auto& h : once.mesh.hn) {
        CHECK(h[1] < h[2]);
        CHECK(h[1] >= 1);  // parents: the shared vertices 1..dim
        CHECK(h[2] <= dim);
    }

    // The neighbour's midpoints on the shared edges reuse the hanging nodes.
    const Index neighbour = dim == 2 ? 4 : 8;  // the unrefined element comes last
    const std::vector<Index> second{neighbour};
    const auto twice = refine(once.mesh, second);
    CHECK(twice.mesh.hn.empty());
    const std::size_t neighbour_edges = dim == 2 ? 3 : 6;
    CHECK(twice.parents.size() == neighbour_edges - shared_edges);
    CHECK(twice.mesh.pos.size() == once.mesh.pos.size() + neighbour_edges - shared_edges);
}

TEST_CASE("Refining nothing leaves the mesh unchanged", "[simplex]") {
    const int dim = GENERATE(2, 3);
    std::mt19937 rng(5);
    const Mesh m = randomly_refined(grid(dim, 3, 0.2, rng), 2, 0.3, rng);

    const auto r = refine(m, std::vector<Index>{});
    CHECK(r.mesh.pos == m.pos);
    CHECK(r.mesh.con == m.con);
    CHECK(r.mesh.hn == m.hn);
    CHECK(r.parents.empty());
}

TEST_CASE("Balanced random refinement keeps nodes, measure and hanging nodes consistent",
          "[simplex]") {
    const int dim = GENERATE(2, 3);
    std::mt19937 rng(static_cast<unsigned>(17 + dim));
    Mesh m = grid(dim, dim == 2 ? 6 : 3, 0.2, rng);
    const double measure = total_measure(m);

    for (int cycle = 0; cycle < 4; ++cycle) {
        const auto r = refine(m, balance_closure(m, random_subset(m.num_elements(), 0.25, rng)));
        const Mesh& out = r.mesh;

        // Old nodes keep their index; each new one is its parents' midpoint, bit-exact.
        REQUIRE(std::equal(m.pos.begin(), m.pos.end(), out.pos.begin()));
        REQUIRE(out.pos.size() == m.pos.size() + r.parents.size());
        for (std::size_t k = 0; k < r.parents.size(); ++k)
            CHECK(out.pos[m.pos.size() + k] ==
                  detail::midpoint(out.pos[static_cast<std::size_t>(r.parents[k][0])],
                                   out.pos[static_cast<std::size_t>(r.parents[k][1])]));

        CHECK_THAT(total_measure(out), WithinRel(measure, 1e-12));
        for (Index e = 0; e < out.num_elements(); ++e)
            CHECK(signed_measure(out, e) > 0);

        // Brute force: the hanging nodes are exactly the nodes sitting at the
        // midpoint of some element edge, with that edge as parents, and none
        // of them hangs on another (1-irregular).
        std::map<Point, Index> node_at;
        for (std::size_t i = 0; i < out.pos.size(); ++i)
            node_at[out.pos[i]] = static_cast<Index>(i);
        std::set<std::array<Index, 3>> expected;
        for (Index e = 0; e < out.num_elements(); ++e) {
            const auto v = out.element(e);
            for (std::size_t a = 0; a < v.size(); ++a)
                for (std::size_t b = a + 1; b < v.size(); ++b) {
                    const auto lo = std::min(v[a], v[b]), hi = std::max(v[a], v[b]);
                    const auto it =
                        node_at.find(detail::midpoint(out.pos[static_cast<std::size_t>(lo)],
                                                      out.pos[static_cast<std::size_t>(hi)]));
                    if (it != node_at.end())
                        expected.insert({it->second, lo, hi});
                }
        }
        const std::set<std::array<Index, 3>> listed(out.hn.begin(), out.hn.end());
        CHECK(listed.size() == out.hn.size());
        CHECK(listed == expected);
        std::set<Index> hanging;
        for (const auto& h : out.hn)
            hanging.insert(h[0]);
        for (const auto& h : out.hn)
            CHECK((!hanging.count(h[1]) && !hanging.count(h[2])));
        m = out;
    }
}

TEST_CASE("Without hanging nodes the closure only sorts and deduplicates", "[simplex]") {
    std::mt19937 rng(3);
    const Mesh m = grid(2, 3, 0.2, rng);
    CHECK(balance_closure(m, std::vector<Index>{5, 1, 5, 0}) == std::vector<Index>{0, 1, 5});
}

TEST_CASE("Balance adds the coarse element across a touched hanging node", "[simplex]") {
    const int dim = GENERATE(2, 3);
    Mesh m;
    m.dim = dim;
    if (dim == 2) {
        m.pos = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}};
        add_element(m, {0, 1, 2});
        add_element(m, {1, 3, 2});
    } else {
        m.pos = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 1}};
        add_element(m, {0, 1, 2, 3});
        add_element(m, {1, 2, 3, 4});
    }
    const Mesh once = refine(m, std::vector<Index>{0}).mesh;
    const Index coarse = once.num_elements() - 1;  // every hanging node sits on its edges
    for (Index child = 0; child < coarse; ++child) {
        const auto closed = balance_closure(once, std::vector<Index>{child});
        const bool touches = std::any_of(once.hn.begin(), once.hn.end(), [&](const auto& h) {
            return detail::contains(once.element(child), h[0]);
        });
        CHECK(std::binary_search(closed.begin(), closed.end(), coarse) == touches);
        CHECK(closed == naive_closure(once, {child}));
    }
}

TEST_CASE("The closure matches the naive fixed point on refined meshes", "[simplex]") {
    const int dim = GENERATE(2, 3);
    std::mt19937 rng(static_cast<unsigned>(29 + dim));
    for (int trial = 0; trial < 5; ++trial) {
        const Mesh m = randomly_refined(grid(dim, dim == 2 ? 4 : 2, 0.2, rng), 3, 0.2, rng);
        REQUIRE(!m.hn.empty());
        for (double fraction : {0.02, 0.1, 0.3}) {
            const auto list = random_subset(m.num_elements(), fraction, rng);
            CHECK(balance_closure(m, list) == naive_closure(m, list));
        }
    }
}

TEST_CASE("Invalid input is rejected", "[simplex]") {
    Mesh m;
    m.pos = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    m.con = {0, 1, 2};
    CHECK_THROWS_AS(refine(m, std::vector<Index>{1}), std::out_of_range);
    CHECK_THROWS_AS(refine(m, std::vector<Index>{-1}), std::out_of_range);
    CHECK_THROWS_AS(balance_closure(m, std::vector<Index>{1}), std::out_of_range);
    Mesh bad_dim = m;
    bad_dim.dim = 4;
    CHECK_THROWS_AS(refine(bad_dim, std::vector<Index>{0}), std::invalid_argument);
    Mesh bad_vertex = m;
    bad_vertex.con = {0, 1, 3};
    CHECK_THROWS_AS(refine(bad_vertex, std::vector<Index>{0}), std::out_of_range);
    Mesh bad_hn = m;
    bad_hn.hn = {{5, 0, 1}};
    CHECK_THROWS_AS(refine(bad_hn, std::vector<Index>{0}), std::out_of_range);
}
