/**
 * @file test_adapt.cpp
 * @brief One solver cycle (adapt.hpp) and the structured seeds (structured.hpp).
 *
 * @details adapt must be exactly marking, then the balance closure, then red
 * refinement, with stagnation decided on the growth. Structured seeds must be
 * conforming simplicial meshes of the box: positively oriented, filling it
 * exactly, and with every facet shared by one element (on the boundary) or two
 * (inside), which is what leaves no hanging node.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <random>
#include <set>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "adapt.hpp"
#include "structured.hpp"
#include "test_util.hpp"

using namespace amr::simplex;
using namespace amr::simplex::test;
using Catch::Matchers::WithinRel;

namespace {

/// Random errors, with about `flag_fraction` of the elements flagged.
void random_indicators(Index n,
                       double flag_fraction,
                       std::mt19937& rng,
                       std::vector<double>& error,
                       std::vector<double>& ratio) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    error.resize(static_cast<std::size_t>(n));
    ratio.resize(static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < error.size(); ++i) {
        error[i] = u(rng);
        ratio[i] = u(rng) < flag_fraction ? 1.5 : 0.5;
    }
}

/// Facets (vertex sets of dim entries) mapped to how many elements have them.
std::map<std::vector<Index>, int> facet_counts(const Mesh& m) {
    std::map<std::vector<Index>, int> count;
    for (Index e = 0; e < m.num_elements(); ++e) {
        const auto v = m.element(e);
        for (std::size_t skip = 0; skip < v.size(); ++skip) {
            std::vector<Index> f;
            for (std::size_t i = 0; i < v.size(); ++i)
                if (i != skip)
                    f.push_back(v[i]);
            std::sort(f.begin(), f.end());
            ++count[f];
        }
    }
    return count;
}

}  // namespace

TEST_CASE("measure is the element's unsigned area or volume", "[adapt]") {
    const int dim = GENERATE(2, 3);
    std::mt19937 rng(5);
    const Mesh m = randomly_refined(grid(dim, 3, 0.2, rng), 2, 0.3, rng);
    const auto length = element_lengths(m);
    for (Index e = 0; e < m.num_elements(); ++e) {
        CHECK_THAT(measure(m, e), WithinRel(std::abs(signed_measure(m, e)), 1e-12));
        CHECK_THAT(length[static_cast<std::size_t>(e)],
                   WithinRel(std::pow(measure(m, e), 1.0 / dim), 1e-15));
    }
}

TEST_CASE("adapt is marking, then balance, then refinement", "[adapt]") {
    const int dim = GENERATE(2, 3);
    std::mt19937 rng(static_cast<unsigned>(71 + dim));
    Mesh m = randomly_refined(grid(dim, dim == 2 ? 5 : 2, 0.2, rng), 2, 0.3, rng);
    for (int cycle = 0; cycle < 3; ++cycle) {
        std::vector<double> error, ratio;
        random_indicators(m.num_elements(), 0.3, rng, error, ratio);
        AdaptParams p;
        p.marking.theta = 0.7;

        const auto a = adapt(m, error, ratio, p);
        REQUIRE(a.status == AdaptStatus::refined);
        const std::vector<Index> growth(static_cast<std::size_t>(m.num_elements()),
                                        dim == 2 ? 3 : 7);
        const auto marking =
            amr::marking::mark(error, ratio, element_lengths(m), growth, p.marking);
        CHECK(a.marking.selected == marking.selected);
        const auto balanced = balance_closure(m, marking.selected);
        CHECK(a.balanced == balanced);
        const auto r = refine(m, balanced);
        CHECK(a.refinement.mesh.pos == r.mesh.pos);
        CHECK(a.refinement.mesh.con == r.mesh.con);
        CHECK(a.refinement.mesh.hn == r.mesh.hn);
        CHECK(a.refinement.parents == r.parents);
        CHECK(std::all_of(a.seconds.begin(), a.seconds.end(), [](double t) { return t >= 0.0; }));
        m = a.refinement.mesh;
    }
}

TEST_CASE("A refinement that grows the mesh too little is stagnated", "[adapt]") {
    std::mt19937 rng(9);
    const Mesh m = grid(2, 4, 0.2, rng);  // 32 triangles
    std::vector<double> error(32, 1.0), ratio(32, 0.5);
    ratio[5] = 2.0;  // one element: +3 elements, a growth of 3/32
    AdaptParams p;
    p.min_growth_fraction = 3.0 / 32.0 + 1e-9;
    CHECK(adapt(m, error, ratio, p).status == AdaptStatus::stagnated);
    p.min_growth_fraction = 3.0 / 32.0;
    CHECK(adapt(m, error, ratio, p).status == AdaptStatus::refined);
    p.min_growth_fraction = 0.0;
    CHECK(adapt(m, error, ratio, p).status == AdaptStatus::refined);
}

TEST_CASE("adapt reports the stop statuses and refines nothing then", "[adapt]") {
    std::mt19937 rng(10);
    const Mesh m = grid(2, 3, 0.2, rng);
    const auto n = static_cast<std::size_t>(m.num_elements());
    std::vector<double> error(n, 1.0), ratio(n, 0.5);
    AdaptParams p;
    auto a = adapt(m, error, ratio, p);
    CHECK(a.status == AdaptStatus::no_candidates);
    CHECK(a.balanced.empty());
    CHECK(a.refinement.mesh.pos.empty());

    ratio.assign(n, 2.0);
    p.marking.min_element_length = 10.0;
    CHECK(adapt(m, error, ratio, p).status == AdaptStatus::floor_exhausted);
    p.marking.min_element_length = 0.0;
    p.marking.max_elements = static_cast<double>(n);
    CHECK(adapt(m, error, ratio, p).status == AdaptStatus::element_ceiling);
    CHECK_THROWS_AS(adapt(m, std::vector<double>(n - 1, 1.0), ratio, p), std::invalid_argument);
}

TEST_CASE("Structured seeds are conforming, oriented simplicial boxes", "[structured]") {
    const std::vector<std::vector<Index>> sizes{
        {2, 2}, {3, 5}, {6, 4}, {7, 7}, {2, 2, 2}, {3, 4, 5}, {5, 3, 2}};
    for (const auto& nodes : sizes) {
        const std::size_t d = nodes.size();
        const std::vector<double> dims =
            d == 2 ? std::vector<double>{554.0, 568.0} : std::vector<double>{554.0, 568.0, 555.0};
        const std::vector<double> origin(d, d == 2 ? 10.0 : -3.5);
        const Mesh m = structured(dims, nodes, origin);
        INFO("nodes " << nodes[0] << "x" << nodes[1]
                      << (d == 3 ? "x" + std::to_string(nodes[2]) : ""));

        Index cells = 1, points = 1;
        for (Index n : nodes) {
            cells *= n - 1;
            points *= n;
        }
        REQUIRE(static_cast<Index>(m.pos.size()) == points);
        REQUIRE(m.num_elements() == cells * (d == 2 ? 2 : 6));

        double box = 1.0;
        for (double x : dims)
            box *= x;
        double total = 0.0;
        for (Index e = 0; e < m.num_elements(); ++e) {
            CHECK(signed_measure(m, e) > 0);
            total += signed_measure(m, e);
        }
        CHECK_THAT(total, WithinRel(box, 1e-12));

        // Every facet is shared by two elements, or lies on the box boundary.
        for (const auto& [facet, count] : facet_counts(m)) {
            REQUIRE((count == 1 || count == 2));
            if (count == 1) {
                bool on_face = false;
                for (std::size_t k = 0; k < d; ++k)
                    for (double side : {origin[k], origin[k] + dims[k]})
                        on_face |= std::all_of(facet.begin(), facet.end(), [&](Index v) {
                            return m.pos[static_cast<std::size_t>(v)][k] == side;
                        });
                CHECK(on_face);
            }
        }
        // Axis ends are exact.
        for (std::size_t k = 0; k < d; ++k) {
            double lo = 1e300, hi = -1e300;
            for (const auto& p : m.pos) {
                lo = std::min(lo, p[k]);
                hi = std::max(hi, p[k]);
            }
            CHECK(lo == origin[k]);
            CHECK(hi == origin[k] + dims[k]);
        }
    }
}

TEST_CASE("The 3D split is the same in every cell", "[structured]") {
    const Mesh m = structured(std::vector<double>{3.0, 4.0, 5.0},
                              std::vector<Index>{4, 5, 6},
                              std::vector<double>{0.0, 0.0, 0.0});
    std::set<std::vector<std::array<int, 3>>> patterns;
    for (Index e = 0; e < m.num_elements(); e += 6) {  // 6 consecutive tetrahedra per cell
        std::array<double, 3> lo{1e300, 1e300, 1e300};
        for (Index t = e; t < e + 6; ++t)
            for (Index v : m.element(t))
                for (std::size_t k = 0; k < 3; ++k)
                    lo[k] = std::min(lo[k], m.pos[static_cast<std::size_t>(v)][k]);
        std::vector<std::array<int, 3>> corners;
        for (Index t = e; t < e + 6; ++t)
            for (Index v : m.element(t)) {
                const auto& p = m.pos[static_cast<std::size_t>(v)];
                corners.push_back({p[0] > lo[0], p[1] > lo[1], p[2] > lo[2]});
            }
        patterns.insert(corners);
    }
    CHECK(patterns.size() == 1);
}

TEST_CASE("2D diagonals follow genMesh's quadrants", "[structured]") {
    // Observed from MATLAB's mesh.genMesh([10 10], nodes): per cell, 1 = the
    // diagonal from the low corner to the high one, 0 = the other; rows are y.
    const std::vector<std::pair<std::vector<Index>, std::vector<std::vector<int>>>> observed{
        {{5, 4}, {{1, 1, 0, 0}, {0, 0, 1, 1}, {0, 0, 1, 1}}},
        {{4, 5}, {{1, 0, 0}, {1, 0, 0}, {0, 1, 1}, {0, 1, 1}}},
        {{6, 6},
         {{1, 1, 0, 0, 0}, {1, 1, 0, 0, 0}, {0, 0, 1, 1, 1}, {0, 0, 1, 1, 1}, {0, 0, 1, 1, 1}}},
    };
    for (const auto& [nodes, diagonal] : observed) {
        const Mesh m =
            structured(std::vector<double>{10.0, 10.0}, nodes, std::vector<double>{0.0, 0.0});
        const double hx = 10.0 / (nodes[0] - 1), hy = 10.0 / (nodes[1] - 1);
        for (Index e = 0; e < m.num_elements(); e += 2) {  // 2 consecutive triangles per cell
            const Index cell = e / 2, j = cell % (nodes[1] - 1),
                        i = cell / (nodes[1] - 1);  // y fastest
            const Point low{i * hx, j * hy, 0.0}, high{(i + 1) * hx, (j + 1) * hy, 0.0};
            bool has_low = false, has_high = false;
            for (Index v : m.element(e)) {
                const auto& p = m.pos[static_cast<std::size_t>(v)];
                has_low |= std::abs(p[0] - low[0]) < 1e-12 && std::abs(p[1] - low[1]) < 1e-12;
                has_high |= std::abs(p[0] - high[0]) < 1e-12 && std::abs(p[1] - high[1]) < 1e-12;
            }
            CHECK(static_cast<int>(has_low && has_high) ==
                  diagonal[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)]);
        }
    }
}

TEST_CASE("linspace stays exact at the ends and finite when the span overflows", "[structured]") {
    const auto y = detail::linspace(-1e308, 1e308, 5);  // b - a overflows
    CHECK(y.front() == -1e308);
    CHECK(y.back() == 1e308);
    CHECK(std::all_of(y.begin(), y.end(), [](double v) { return std::isfinite(v); }));
    CHECK(std::is_sorted(y.begin(), y.end()));
    const auto z = detail::linspace(0.0, 1.5e308, 4);  // (b - a) * (n - 2) overflows
    CHECK(std::all_of(z.begin(), z.end(), [](double v) { return std::isfinite(v); }));
    CHECK(std::is_sorted(z.begin(), z.end()));
}

TEST_CASE("Structured input is validated", "[structured]") {
    const std::vector<double> zero2{0.0, 0.0};
    CHECK_THROWS_AS(structured(std::vector<double>{-1.0, 1.0}, std::vector<Index>{2, 2}, zero2),
                    std::invalid_argument);
    CHECK_THROWS_AS(
        structured(std::vector<double>{std::nan(""), 1.0}, std::vector<Index>{2, 2}, zero2),
        std::invalid_argument);
    CHECK_THROWS_AS(
        structured(std::vector<double>{1.0, 1.0}, std::vector<Index>{70000, 70000}, zero2),
        std::overflow_error);
    CHECK_THROWS_AS(
        structured(std::vector<double>{1.0}, std::vector<Index>{2}, std::vector<double>{0.0}),
        std::invalid_argument);
    CHECK_THROWS_AS(
        structured(
            std::vector<double>{1.0, 1.0}, std::vector<Index>{2, 1}, std::vector<double>{0.0, 0.0}),
        std::invalid_argument);
    CHECK_THROWS_AS(
        structured(
            std::vector<double>{1.0, 1.0}, std::vector<Index>{2, 2}, std::vector<double>{0.0}),
        std::invalid_argument);
}
