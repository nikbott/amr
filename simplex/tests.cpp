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
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "simplex.hpp"

using namespace amr::simplex;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

Point sub(const Point& a, const Point& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

/// Signed area (T3) or volume (T4) of element e.
double signed_measure(const Mesh& m, Index e) {
    const auto v = m.element(e);
    const auto p = [&](std::size_t i) { return m.pos[static_cast<std::size_t>(v[i])]; };
    const Point u = sub(p(1), p(0)), w = sub(p(2), p(0));
    if (m.dim == 2)
        return 0.5 * (u[0] * w[1] - u[1] * w[0]);
    const Point x = sub(p(3), p(0));
    return (u[0] * (w[1] * x[2] - w[2] * x[1]) - u[1] * (w[0] * x[2] - w[2] * x[0]) +
            u[2] * (w[0] * x[1] - w[1] * x[0])) /
           6.0;
}

double total_measure(const Mesh& m) {
    double sum = 0;
    for (Index e = 0; e < m.num_elements(); ++e)
        sum += signed_measure(m, e);
    return sum;
}

/// Positively oriented element from its vertices.
void add_element(Mesh& m, std::vector<Index> v) {
    const auto first = m.con.size();
    m.con.insert(m.con.end(), v.begin(), v.end());
    if (signed_measure(m, m.num_elements() - 1) < 0)
        std::swap(m.con[first], m.con[first + 1]);
}

/// n^dim grid of unit cells, 2 triangles or 6 (Kuhn) tetrahedra per cell,
/// interior nodes jittered so no two diagonals tie.
Mesh grid(int dim, int n, double jitter, std::mt19937& rng) {
    Mesh m;
    m.dim = dim;
    const int nz = dim == 3 ? n : 0;
    std::uniform_real_distribution<double> u(-jitter, jitter);
    const auto id = [n](int i, int j, int k) {
        return static_cast<Index>((k * (n + 1) + j) * (n + 1) + i);
    };
    for (int k = 0; k <= nz; ++k)
        for (int j = 0; j <= n; ++j)
            for (int i = 0; i <= n; ++i) {
                const bool interior =
                    i > 0 && i < n && j > 0 && j < n && (dim == 2 || (k > 0 && k < n));
                m.pos.push_back({i + (interior ? u(rng) : 0.0),
                                 j + (interior ? u(rng) : 0.0),
                                 dim == 3 ? k + (interior ? u(rng) : 0.0) : 0.0});
            }
    for (int k = 0; k < std::max(nz, 1); ++k)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                if (dim == 2) {
                    add_element(m, {id(i, j, 0), id(i + 1, j, 0), id(i + 1, j + 1, 0)});
                    add_element(m, {id(i, j, 0), id(i + 1, j + 1, 0), id(i, j + 1, 0)});
                    continue;
                }
                std::array<int, 3> axes{0, 1, 2};
                do {  // Kuhn: one tetrahedron per path from (i,j,k) to the far corner
                    std::array<int, 3> c{i, j, k};
                    std::vector<Index> v{id(c[0], c[1], c[2])};
                    for (int a : axes) {
                        ++c[static_cast<std::size_t>(a)];
                        v.push_back(id(c[0], c[1], c[2]));
                    }
                    add_element(m, v);
                } while (std::next_permutation(axes.begin(), axes.end()));
            }
    return m;
}

std::vector<Index> random_subset(Index n, double fraction, std::mt19937& rng) {
    std::vector<Index> all(static_cast<std::size_t>(n));
    std::iota(all.begin(), all.end(), 0);
    std::shuffle(all.begin(), all.end(), rng);
    all.resize(static_cast<std::size_t>(std::ceil(fraction * n)));
    return all;
}

/// The MATLAB loop, verbatim: sweep the hanging nodes until nothing changes.
std::vector<Index> naive_closure(const Mesh& m, const std::vector<Index>& list) {
    std::vector<char> sel(static_cast<std::size_t>(m.num_elements()), 0);
    for (Index e : list)
        sel[static_cast<std::size_t>(e)] = 1;
    const auto has = [&m](Index e, Index v) { return detail::contains(m.element(e), v); };
    for (bool changed = true; changed;) {
        changed = false;
        for (const auto& h : m.hn) {
            bool touched = false;
            for (Index e = 0; e < m.num_elements(); ++e)
                touched |= sel[static_cast<std::size_t>(e)] && has(e, h[0]);
            if (!touched)
                continue;
            for (Index e = 0; e < m.num_elements(); ++e)
                if (!sel[static_cast<std::size_t>(e)] && has(e, h[1]) && has(e, h[2]) &&
                    !has(e, h[0])) {
                    sel[static_cast<std::size_t>(e)] = 1;
                    changed = true;
                }
        }
    }
    std::vector<Index> out;
    for (Index e = 0; e < m.num_elements(); ++e)
        if (sel[static_cast<std::size_t>(e)])
            out.push_back(e);
    return out;
}

/// `cycles` rounds of refining a balanced random fraction of the elements.
Mesh randomly_refined(Mesh m, int cycles, double fraction, std::mt19937& rng) {
    for (int c = 0; c < cycles; ++c)
        m = refine(m, balance_closure(m, random_subset(m.num_elements(), fraction, rng))).mesh;
    return m;
}

/// Worst child quality of the best octahedron split, enumerating the three
/// diagonals (midpoints of opposite edges) and the equator around each.
double best_octahedron_worst_quality(const std::array<Point, 4>& c) {
    std::map<std::pair<int, int>, Point> mid;
    for (int a = 0; a < 4; ++a)
        for (int b = a + 1; b < 4; ++b)
            mid[{a, b}] =
                detail::midpoint(c[static_cast<std::size_t>(a)], c[static_cast<std::size_t>(b)]);
    const auto shares = [](std::pair<int, int> x, std::pair<int, int> y) {
        return x.first == y.first || x.first == y.second || x.second == y.first ||
               x.second == y.second;
    };
    double best = -1;
    for (const auto& [d0, d1] : std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>>{
             {{0, 1}, {2, 3}}, {{0, 2}, {1, 3}}, {{0, 3}, {1, 2}}}) {
        std::vector<std::pair<int, int>> ring;
        for (const auto& [e, p] : mid)
            if (e != d0 && e != d1)
                ring.push_back(e);
        for (std::size_t i = 1; i < ring.size(); ++i)  // order the equator into a cycle
            for (std::size_t j = i; j < ring.size(); ++j)
                if (shares(ring[i - 1], ring[j])) {
                    std::swap(ring[i], ring[j]);
                    break;
                }
        double worst = 1e300;
        for (std::size_t i = 0; i < 4; ++i)
            worst = std::min(
                worst, detail::tet_quality(mid[d0], mid[d1], mid[ring[i]], mid[ring[(i + 1) % 4]]));
        best = std::max(best, worst);
    }
    return best;
}

}  // namespace

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
        for (Index e = 4; e < 8; ++e) {  // children 4..7 fill the octahedron
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
