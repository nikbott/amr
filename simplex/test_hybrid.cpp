/**
 * @file test_hybrid.cpp
 * @brief Refinement, balance and adapt on hybrid meshes of every shape.
 *
 * @details The oracle is test_util's conformity_errors(): continuity of the
 * constrained field across every face, an exact tiling of the box, and the
 * hanging set derived from the geometry alone. It never reads the templates,
 * so it checks them too: a face refined differently from its two sides, a
 * missing or misplaced constraint, or a hole between children all show up.
 */

#include <algorithm>
#include <random>
#include <set>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "adapt.hpp"
#include "simplex.hpp"
#include "test_util.hpp"

using namespace amr::simplex;
using namespace amr::simplex::test;
using Catch::Matchers::WithinRel;

TEST_CASE("Hybrid seeds are conforming and use every shape", "[hybrid]") {
    std::mt19937 rng(3);
    const Mesh m2 = hybrid_2d(4, rng), m3 = hybrid_3d(3, rng);
    CHECK(conformity_errors(m2, {0, 0, 0}, {8, 8, 0}, 2000, rng).empty());
    CHECK(conformity_errors(m3, {0, 0, 0}, {6, 6, 8}, 2000, rng).empty());
    CHECK(std::set<Shape>(m2.type.begin(), m2.type.end()) == std::set<Shape>{Shape::T3, Shape::Q4});
    CHECK(std::set<Shape>(m3.type.begin(), m3.type.end()) ==
          std::set<Shape>{Shape::T4, Shape::H8, Shape::W6, Shape::P5});
}

TEST_CASE("Balanced refinement keeps hybrid meshes conforming", "[hybrid]") {
    const int dim = GENERATE(2, 3);
    const auto seed = static_cast<unsigned>(GENERATE(1, 2, 3));
    std::mt19937 rng(seed);
    Mesh m = dim == 2 ? hybrid_2d(3, rng) : hybrid_3d(2, rng);
    const Point hi = dim == 2 ? Point{6, 6, 0} : Point{4, 4, 8};
    const double volume = total_measure(m);
    std::set<Shape> refined;
    for (int cycle = 0; cycle < 3; ++cycle) {
        const auto list = balance_closure(m, random_subset(m.num_elements(), 0.3, rng));
        for (Index e : list)
            refined.insert(m.type[static_cast<std::size_t>(e)]);
        const auto r = refine(m, list);
        INFO("dim " << dim << ", seed " << seed << ", cycle " << cycle);
        CHECK(conformity_errors(r.mesh, {0, 0, 0}, hi, 1000, rng).empty());
        CHECK_THAT(total_measure(r.mesh), WithinRel(volume, 1e-12));
        // The prolongation is the new nodes' constraints: rows in creation
        // order, at their parents' weighted mean.
        REQUIRE(r.mesh.pos.size() == m.pos.size() + r.prolongation.size());
        for (std::size_t k = 0; k < r.prolongation.size(); ++k)
            CHECK(r.prolongation[k].node == static_cast<Index>(m.pos.size() + k));
        m = r.mesh;
    }
    CHECK(refined.size() == (dim == 2 ? 2u : 4u));
}

TEST_CASE("The closure matches the naive fixed point on hybrid meshes", "[hybrid]") {
    const int dim = GENERATE(2, 3);
    std::mt19937 rng(static_cast<unsigned>(37 + dim));
    for (int trial = 0; trial < 4; ++trial) {
        const Mesh m =
            randomly_refined(dim == 2 ? hybrid_2d(3, rng) : hybrid_3d(2, rng), 2, 0.2, rng);
        REQUIRE(std::any_of(m.hn.begin(), m.hn.end(), [](const auto& h) {
                    return h.parent.size() == 4;  // face centres hang in 3D, only midpoints in 2D
                }) == (dim == 3));
        for (double fraction : {0.02, 0.1, 0.3}) {
            const auto list = random_subset(m.num_elements(), fraction, rng);
            CHECK(balance_closure(m, list) == naive_closure(m, list));
        }
    }
}

TEST_CASE("A quadrilateral face's nodes are shared, whichever side refines first", "[hybrid]") {
    // A hexahedron on top of a pyramid: their shared face is the hexahedron's
    // bottom and the pyramid's base.
    LatticeMesh g;
    g.mesh.dim = 3;
    const auto c = [&](int x, int y, int z) { return g.at(2 * x, 2 * y, 2 * z); };
    add_oriented(g.mesh,
                 Shape::H8,
                 {c(0, 0, 1),
                  c(1, 0, 1),
                  c(1, 1, 1),
                  c(0, 1, 1),
                  c(0, 0, 2),
                  c(1, 0, 2),
                  c(1, 1, 2),
                  c(0, 1, 2)});
    add_oriented(
        g.mesh, Shape::P5, {c(0, 0, 1), c(1, 0, 1), c(1, 1, 1), c(0, 1, 1), g.at(1, 1, 0)});
    const Mesh m = g.mesh;
    const double volume = total_measure(m);
    std::mt19937 rng(5);
    std::set<std::vector<Point>> finals;
    for (const Index first : {0, 1}) {
        const auto once = refine(m, std::vector<Index>{first});
        // The face centre (4 parents) and the 4 midpoints of the face's edges hang.
        CHECK(once.mesh.hn.size() == 5);
        CHECK(std::count_if(once.mesh.hn.begin(), once.mesh.hn.end(), [](const auto& h) {
                  return h.parent.size() == 4 && h.weight == std::vector<double>(4, 0.25);
              }) == 1);
        // The pair does not fill a box, so no tiling samples; the measure instead.
        CHECK(conformity_errors(once.mesh, {}, {}, 0, rng).empty());
        CHECK_THAT(total_measure(once.mesh), WithinRel(volume, 1e-12));
        const Index other = once.mesh.num_elements() - 1;
        const auto twice = refine(once.mesh, std::vector<Index>{other});
        CHECK(twice.mesh.hn.empty());
        CHECK(conformity_errors(twice.mesh, {}, {}, 0, rng).empty());
        CHECK_THAT(total_measure(twice.mesh), WithinRel(volume, 1e-12));
        // The second element reuses all 5 nodes of the shared face.
        const auto& t = element_type(m.type[static_cast<std::size_t>(first == 0 ? 1 : 0)]);
        CHECK(twice.prolongation.size() == t.red.new_nodes.size() - 5);
        std::vector<Point> p = twice.mesh.pos;
        std::sort(p.begin(), p.end());
        finals.insert(p);
    }
    CHECK(finals.size() == 1);  // the same nodes either way
}

TEST_CASE("adapt counts each shape's children in its ceiling and stagnation", "[hybrid]") {
    std::mt19937 rng(8);
    const Mesh m = hybrid_3d(1, rng);
    const auto n = static_cast<std::size_t>(m.num_elements());
    for (const Shape s : {Shape::T4, Shape::W6, Shape::P5}) {
        const auto e =
            static_cast<std::size_t>(std::find(m.type.begin(), m.type.end(), s) - m.type.begin());
        REQUIRE(e < n);
        std::vector<double> error(n, 1.0), ratio(n, 0.5);
        ratio[e] = 2.0;  // only e is flagged; nothing hangs, so the closure is {e}
        const auto added = static_cast<double>(element_type(s).red.children() - 1);
        AdaptParams p;
        p.marking.max_elements = static_cast<double>(n) + added;
        const auto a = adapt(m, error, ratio, p);
        REQUIRE(a.status == AdaptStatus::refined);
        CHECK(static_cast<double>(a.refinement.mesh.num_elements()) ==
              static_cast<double>(n) + added);
        p.marking.max_elements -= 1;
        CHECK(adapt(m, error, ratio, p).status == AdaptStatus::element_ceiling);
        p.marking.max_elements = 0;
        p.min_growth_fraction = added / static_cast<double>(n) + 1e-9;
        CHECK(adapt(m, error, ratio, p).status == AdaptStatus::stagnated);
    }
}

TEST_CASE("measure is each shape's unsigned area or volume", "[hybrid]") {
    std::mt19937 rng(9);
    for (const Mesh& m : {randomly_refined(hybrid_2d(3, rng), 2, 0.3, rng),
                          randomly_refined(hybrid_3d(2, rng), 1, 0.3, rng)}) {
        double total = 0;
        for (Index e = 0; e < m.num_elements(); ++e) {
            CHECK_THAT(measure(m, e), WithinRel(signed_measure(m, e), 1e-15));
            total += measure(m, e);
        }
        CHECK_THAT(total, WithinRel(m.dim == 2 ? 36.0 : 128.0, 1e-12));
    }
}
