/**
 * @file test_element.cpp
 * @brief The reference-element model (element.hpp) against its references.
 *
 * @details Freudenthal's subdivision [Freudenthal1942] must tile the d-simplex
 * with 2^d children of equal volume; under repeated refinement in its own
 * vertex order it must produce at most d!/2 congruence classes [Bey2000]. The
 * solver's T3/T4 orders must list exactly the generated subdivisions. Every
 * shape's children must be positively oriented, split the measure as red
 * refinement does, reproduce the shape's multilinear fields at the new nodes,
 * and induce on each face exactly that face's own refinement, which is what
 * makes neighbours of any shape conforming.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "element.hpp"
#include "simplex.hpp"
#include "structured.hpp"

using namespace amr::simplex;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

const std::vector<Shape> kShapes{Shape::T3, Shape::Q4, Shape::T4, Shape::H8, Shape::W6, Shape::P5};

/// Reference vertex coordinates, in each shape's node order.
std::vector<Point> reference(Shape s) {
    switch (s) {
        case Shape::T3:
            return {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
        case Shape::Q4:
            return {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
        case Shape::T4:
            return {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        case Shape::H8:
            return {{0, 0, 0},
                    {1, 0, 0},
                    {1, 1, 0},
                    {0, 1, 0},
                    {0, 0, 1},
                    {1, 0, 1},
                    {1, 1, 1},
                    {0, 1, 1}};
        case Shape::W6:
            return {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {0, 1, 1}};
        case Shape::P5:
            return {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0.5, 0.5, 1}};
    }
    return {};
}

/// A random affine map with positive determinant (in-plane for 2D shapes).
struct Affine {
    std::array<std::array<double, 3>, 3> a{};
    Point b{};
    Point operator()(const Point& p) const {
        Point q = b;
        for (std::size_t r = 0; r < 3; ++r)
            for (std::size_t c = 0; c < 3; ++c)
                q[r] += a[r][c] * p[c];
        return q;
    }
};

Affine random_affine(int dim, std::mt19937& rng) {
    std::uniform_real_distribution<double> u(-0.4, 0.4);
    Affine f;
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            f.a[r][c] = (r == c ? 1.5 : 0.0) + (dim == 2 && (r == 2 || c == 2) ? 0.0 : u(rng));
    if (dim == 2)
        f.a[2][2] = 1.0;
    for (std::size_t r = 0; r < (dim == 2 ? 2u : 3u); ++r)
        f.b[r] = u(rng) * 10;
    return f;
}

/// Positions of the parent's vertices and new nodes, in local order.
std::vector<Point> local_nodes(const ElementType& t, const std::vector<Point>& vertices) {
    std::vector<Point> x = vertices;
    for (const auto& n : t.red.new_nodes) {
        Point p{0, 0, 0};
        for (std::size_t i = 0; i < n.vertex.size(); ++i)
            for (std::size_t c = 0; c < 3; ++c)
                p[c] += n.weight[i] * vertices[static_cast<std::size_t>(n.vertex[i])][c];
        x.push_back(p);
    }
    return x;
}

std::vector<Child> all_children(const RedTemplate& red, std::size_t choice) {
    std::vector<Child> c = red.head;
    c.insert(c.end(), red.choices[choice].begin(), red.choices[choice].end());
    c.insert(c.end(), red.tail.begin(), red.tail.end());
    return c;
}

std::vector<Point> positions(const std::vector<Point>& local, const std::vector<int>& nodes) {
    std::vector<Point> p;
    for (int n : nodes)
        p.push_back(local[static_cast<std::size_t>(n)]);
    return p;
}

/// Barycentric point (x2) -> parent-local node, for simplex templates.
std::vector<int> barycentric_of(const ElementType& t, int local) {
    std::vector<int> b(static_cast<std::size_t>(t.vertices), 0);
    if (local < t.vertices) {
        b[static_cast<std::size_t>(local)] = 2;
        return b;
    }
    const auto& n = t.red.new_nodes[static_cast<std::size_t>(local - t.vertices)];
    for (std::size_t i = 0; i < n.vertex.size(); ++i)
        b[static_cast<std::size_t>(n.vertex[i])] = static_cast<int>(std::lround(2 * n.weight[i]));
    return b;
}

using ChildSet = std::set<std::set<std::vector<int>>>;

ChildSet as_barycentric(const ElementType& t, const std::vector<Child>& children) {
    ChildSet out;
    for (const auto& c : children) {
        std::set<std::vector<int>> s;
        for (int n : c.node)
            s.insert(barycentric_of(t, n));
        out.insert(s);
    }
    return out;
}

ChildSet as_barycentric(const detail::SimplexSplit& s) {
    ChildSet out;
    for (const auto& c : s.child) {
        std::set<std::vector<int>> v;
        for (int i : c)
            v.insert(s.point[static_cast<std::size_t>(i)]);
        out.insert(v);
    }
    return out;
}

/// Congruence class key of a tetrahedron: its 6 edge lengths under the vertex
/// permutation that makes the tuple smallest (relative rounding 1e-9).
std::vector<long long> congruence_key(const std::array<Point, 4>& t) {
    const std::array<std::array<int, 2>, 6> e{{{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}};
    std::array<int, 4> p{0, 1, 2, 3};
    std::vector<long long> best;
    do {
        std::vector<long long> key;
        for (const auto& [a, b] : e) {
            const Point d =
                detail::sub(t[static_cast<std::size_t>(p[static_cast<std::size_t>(a)])],
                            t[static_cast<std::size_t>(p[static_cast<std::size_t>(b)])]);
            key.push_back(std::llround(std::sqrt(detail::dot(d, d)) * 1e9));
        }
        if (best.empty() || key < best)
            best = key;
    } while (std::next_permutation(p.begin(), p.end()));
    return best;
}

}  // namespace

TEST_CASE("Freudenthal tiles the d-simplex with 2^d equal children", "[element]") {
    std::mt19937 rng(3);
    for (int d = 1; d <= 4; ++d) {
        std::vector<int> order(static_cast<std::size_t>(d + 1));
        std::iota(order.begin(), order.end(), 0);
        const auto s = detail::freudenthal(d, order);
        INFO("d = " << d);
        REQUIRE(s.child.size() == (std::size_t{1} << d));
        for (const auto& c : s.child) {
            std::vector<std::vector<long long>> b;
            for (int i : c)
                b.emplace_back(s.point[static_cast<std::size_t>(i)].begin(),
                               s.point[static_cast<std::size_t>(i)].end());
            // |det| of the child's barycentric rows (x2) is 2^(d+1) / 2^d = 2 for a
            // child of volume 1/2^d; positive: the parent's orientation. The sign
            // is exact and Bareiss keeps the magnitude integral.
            CHECK(detail::det_sign(b) > 0);
        }
        // Every interior point lies in exactly one child: a tiling.
        std::uniform_real_distribution<double> u(0.0, 1.0);
        for (int trial = 0; trial < 300; ++trial) {
            std::vector<double> lambda(static_cast<std::size_t>(d + 1));
            double sum = 0;
            for (auto& l : lambda)
                sum += (l = -std::log(u(rng) + 1e-300));  // uniform on the simplex
            for (auto& l : lambda)
                l /= sum;
            int inside = 0;
            for (const auto& c : s.child) {
                // Solve lambda = sum_k mu_k * point_k / 2 for the child weights mu.
                std::vector<std::vector<double>> a(
                    static_cast<std::size_t>(d + 1),
                    std::vector<double>(static_cast<std::size_t>(d + 2)));
                for (std::size_t r = 0; r <= static_cast<std::size_t>(d); ++r) {
                    for (std::size_t k = 0; k <= static_cast<std::size_t>(d); ++k)
                        a[r][k] = s.point[static_cast<std::size_t>(c[k])][r] / 2.0;
                    a[r][static_cast<std::size_t>(d + 1)] = lambda[r];
                }
                for (std::size_t k = 0; k <= static_cast<std::size_t>(d); ++k) {  // Gauss-Jordan
                    std::size_t piv = k;
                    for (std::size_t r = k; r <= static_cast<std::size_t>(d); ++r)
                        if (std::abs(a[r][k]) > std::abs(a[piv][k]))
                            piv = r;
                    std::swap(a[k], a[piv]);
                    for (std::size_t r = 0; r <= static_cast<std::size_t>(d); ++r)
                        if (r != k) {
                            const double f = a[r][k] / a[k][k];
                            for (std::size_t j = k; j <= static_cast<std::size_t>(d + 1); ++j)
                                a[r][j] -= f * a[k][j];
                        }
                }
                bool in = true;
                for (std::size_t k = 0; k <= static_cast<std::size_t>(d); ++k)
                    in &= a[k][static_cast<std::size_t>(d + 1)] / a[k][k] > 1e-12;
                inside += in;
            }
            CHECK(inside == 1);
        }
    }
}

TEST_CASE("A tetrahedron has 3 Freudenthal subdivisions, a triangle 1", "[element]") {
    CHECK(detail::freudenthal_candidates(2).size() == 1);
    CHECK(detail::freudenthal_candidates(3).size() == 3);
}

TEST_CASE("Repeated Freudenthal refinement keeps at most d!/2 congruence classes", "[element]") {
    // [Bey2000]: refining in each child's own Kuhn vertex order, a tetrahedron's
    // descendants fall into at most 3! / 2 = 3 congruence classes.
    std::vector<int> order{0, 1, 2, 3};
    const auto split = detail::freudenthal(3, order, /*oriented=*/false);
    std::mt19937 rng(5);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<std::array<Point, 4>> level{{{{0, 0, 0},
                                              {1, 0.1 * u(rng), 0.2 * u(rng)},
                                              {0.3 * u(rng), 1, 0.1},
                                              {0.2, 0.3 * u(rng), 1}}}};
    for (int generation = 0; generation < 3; ++generation) {
        std::vector<std::array<Point, 4>> next;
        for (const auto& t : level)
            for (const auto& c : split.child) {
                std::array<Point, 4> child{};
                for (std::size_t k = 0; k < 4; ++k) {
                    const auto& b = split.point[static_cast<std::size_t>(c[k])];
                    for (std::size_t v = 0; v < 4; ++v)
                        for (std::size_t x = 0; x < 3; ++x)
                            child[k][x] += b[v] / 2.0 * t[v][x];
                }
                next.push_back(child);
            }
        level = std::move(next);
    }
    REQUIRE(level.size() == 512);
    std::set<std::vector<long long>> classes;
    for (const auto& t : level)
        classes.insert(congruence_key(t));
    CHECK(classes.size() <= 3);
}

TEST_CASE("The solver's T3 and T4 orders list exactly the generated subdivisions", "[element]") {
    const auto& t3 = element_type(Shape::T3);
    CHECK(as_barycentric(t3, all_children(t3.red, 0)) ==
          as_barycentric(detail::freudenthal_candidates(2)[0]));

    const auto& t4 = element_type(Shape::T4);
    const auto generated = detail::freudenthal_candidates(3);
    REQUIRE(t4.red.choices.size() == generated.size());
    std::set<ChildSet> legacy, model;
    for (std::size_t k = 0; k < generated.size(); ++k) {
        legacy.insert(as_barycentric(t4, all_children(t4.red, k)));
        model.insert(as_barycentric(generated[k]));
    }
    CHECK(legacy == model);
    // Choice 0 is the identity ordering's split, Bey's.
    CHECK(as_barycentric(t4, all_children(t4.red, 0)) == as_barycentric(generated[0]));
}

TEST_CASE("Children are oriented, split the measure and cover the parent", "[element]") {
    std::mt19937 rng(11);
    for (Shape s : kShapes) {
        const auto& t = element_type(s);
        INFO("shape " << static_cast<int>(s));
        for (int trial = 0; trial < 20; ++trial) {
            const auto f = random_affine(t.dim, rng);
            std::vector<Point> v;
            for (const auto& p : reference(s))
                v.push_back(f(p));
            const double parent = signed_measure(s, v);
            REQUIRE(parent > 0);
            const auto local = local_nodes(t, v);
            for (std::size_t choice = 0; choice < t.red.choices.size(); ++choice) {
                const auto children = all_children(t.red, choice);
                double total = 0;
                for (const auto& c : children) {
                    const double m = signed_measure(c.shape, positions(local, c.node));
                    CHECK(m > 0);
                    // Red refinement halves every length: children of the parent's
                    // shape have 1/2^dim of its measure (the pyramid's tetrahedra 1/16).
                    const double expected = parent / (c.shape == s ? (1 << t.dim) : 16);
                    CHECK_THAT(m, WithinRel(expected, 1e-9));
                    total += m;
                }
                CHECK_THAT(total, WithinRel(parent, 1e-12));
            }
        }
    }
}

TEST_CASE("New nodes reproduce the parent's multilinear fields", "[element]") {
    // A node's weights are the parent's shape functions at it, so a field in
    // the element's space is reproduced exactly: linear in x, y, z for
    // simplices, plus the cross terms of each product.
    std::mt19937 rng(13);
    std::uniform_real_distribution<double> u(-1, 1);
    for (Shape s : {Shape::T3, Shape::Q4, Shape::T4, Shape::H8, Shape::W6}) {
        const auto& t = element_type(s);
        const auto ref = reference(s);
        const std::array<double, 8> c{
            u(rng), u(rng), u(rng), u(rng), u(rng), u(rng), u(rng), u(rng)};
        const auto field = [&](const Point& p) {
            double v = c[0] + c[1] * p[0] + c[2] * p[1] + c[3] * p[2];
            if (s == Shape::Q4 || s == Shape::H8)
                v += c[4] * p[0] * p[1];
            if (s == Shape::H8)
                v += c[5] * p[0] * p[2] + c[6] * p[1] * p[2] + c[7] * p[0] * p[1] * p[2];
            if (s == Shape::W6)
                v += c[5] * p[0] * p[2] + c[6] * p[1] * p[2];
            return v;
        };
        const auto local = local_nodes(t, ref);
        INFO("shape " << static_cast<int>(s));
        for (std::size_t k = 0; k < t.red.new_nodes.size(); ++k) {
            const auto& n = t.red.new_nodes[k];
            double interpolated = 0;
            for (std::size_t i = 0; i < n.vertex.size(); ++i)
                interpolated += n.weight[i] * field(ref[static_cast<std::size_t>(n.vertex[i])]);
            CHECK_THAT(interpolated,
                       WithinAbs(field(local[static_cast<std::size_t>(t.vertices) + k]), 1e-12));
            CHECK_THAT(std::accumulate(n.weight.begin(), n.weight.end(), 0.0),
                       WithinAbs(1.0, 1e-15));
        }
    }
}

TEST_CASE("Children induce each face's own refinement", "[element]") {
    // A face shared by two elements of any shapes is refined the same way from
    // both sides, so red refinement leaves hybrid meshes conforming.
    for (Shape s : kShapes) {
        const auto& t = element_type(s);
        INFO("shape " << static_cast<int>(s));
        // Lattice position of a local node, as parent-vertex weights.
        const auto weights = [&](int n) {
            std::map<int, double> w;
            if (n < t.vertices)
                w[n] = 1.0;
            else {
                const auto& node = t.red.new_nodes[static_cast<std::size_t>(n - t.vertices)];
                for (std::size_t i = 0; i < node.vertex.size(); ++i)
                    w[node.vertex[i]] = node.weight[i];
            }
            return w;
        };
        const auto facets = t.dim == 2 ? [&] {
            std::vector<std::vector<int>> e;
            for (const auto& [a, b] : t.edges)
                e.push_back({a, b});
            return e;
        }()
                                       : t.faces;
        for (std::size_t choice = 0; choice < t.red.choices.size(); ++choice)
            for (const auto& facet : facets) {
                const std::set<int> fv(facet.begin(), facet.end());
                // The children's facets lying on this facet, as weight maps.
                std::set<std::set<std::map<int, double>>> actual;
                for (const auto& c : all_children(t.red, choice)) {
                    const auto& ct = element_type(c.shape);
                    std::vector<std::vector<int>> cf;
                    if (ct.dim == 2)
                        for (const auto& [a, b] : ct.edges)
                            cf.push_back({a, b});
                    else
                        cf = ct.faces;
                    for (const auto& f : cf) {
                        std::set<std::map<int, double>> nodes;
                        bool on = true;
                        for (int k : f) {
                            const auto w = weights(c.node[static_cast<std::size_t>(k)]);
                            for (const auto& [v, x] : w)
                                on &= fv.contains(v);
                            nodes.insert(w);
                        }
                        if (on)
                            actual.insert(nodes);
                    }
                }
                // The facet's own red refinement, mapped onto the parent's vertices.
                const Shape fs =
                    facet.size() == 2 ? Shape::T3 : (facet.size() == 3 ? Shape::T3 : Shape::Q4);
                std::set<std::set<std::map<int, double>>> expected;
                if (facet.size() == 2) {  // an edge: two halves
                    std::map<int, double> a{{facet[0], 1.0}}, b{{facet[1], 1.0}},
                        m{{facet[0], 0.5}, {facet[1], 0.5}};
                    expected = {{a, m}, {m, b}};
                } else {
                    const auto& cyc = facet;  // cyclic, as the face's own element orders it
                    const auto& ft = element_type(fs);
                    for (const auto& fc : all_children(ft.red, 0)) {
                        std::set<std::map<int, double>> nodes;
                        for (int k : fc.node) {
                            std::map<int, double> w;
                            if (k < ft.vertices)
                                w[cyc[static_cast<std::size_t>(k)]] = 1.0;
                            else {
                                const auto& n =
                                    ft.red.new_nodes[static_cast<std::size_t>(k - ft.vertices)];
                                for (std::size_t i = 0; i < n.vertex.size(); ++i)
                                    w[cyc[static_cast<std::size_t>(n.vertex[i])]] += n.weight[i];
                            }
                            nodes.insert(w);
                        }
                        expected.insert(nodes);
                    }
                }
                CHECK(actual == expected);
            }
    }
}

TEST_CASE("The split choice takes the best worst child", "[element]") {
    std::mt19937 rng(17);
    std::uniform_real_distribution<double> u(0, 1);
    const auto& t = element_type(Shape::T4);
    for (int trial = 0; trial < 200; ++trial) {
        std::vector<Point> v{{0, 0, 0},
                             {1, u(rng), u(rng) * 0.3},
                             {u(rng) * 0.3, 1, u(rng)},
                             {u(rng), u(rng) * 0.3, 1}};
        if (signed_measure(Shape::T4, v) <= 0)
            std::swap(v[0], v[1]);
        const auto local = local_nodes(t, v);
        const auto chosen = split_choice(t.red, local);
        const auto worst = [&](std::size_t k) {
            double w = 1e300;
            for (const auto& c : t.red.choices[k])
                w = std::min(w, simplex_quality(Shape::T4, positions(local, c.node)));
            return w;
        };
        for (std::size_t k = 0; k < t.red.choices.size(); ++k)
            CHECK(worst(chosen) * (1.0 + kSplitTieTolerance) >= worst(k));
    }
}

TEST_CASE("Tied splits go to the earliest choice, whatever the round-off", "[element]") {
    // A structured seed of cube cells with inexact coordinates (512/3
    // spacing): a third of its tetrahedra tie in exact arithmetic, and
    // round-off used to give 24 of these 54 ties to a later split. The
    // earliest choice within the tie tolerance must win.
    const Mesh m = structured(std::vector<double>{512, 512, 512},
                              std::vector<Index>{4, 4, 4},
                              std::vector<double>{0, 0, 0});
    const auto& t = element_type(Shape::T4);
    int ties = 0;
    for (Index e = 0; e < m.num_elements(); ++e) {
        std::vector<Point> v;
        for (Index n : m.element(e))
            v.push_back(m.pos[static_cast<std::size_t>(n)]);
        const auto local = local_nodes(t, v);
        std::vector<double> w;
        for (const auto& choice : t.red.choices) {
            double worst = 1e300;
            for (const auto& c : choice)
                worst = std::min(worst, simplex_quality(Shape::T4, positions(local, c.node)));
            w.push_back(worst);
        }
        const double best = *std::max_element(w.begin(), w.end());
        std::size_t earliest = 0;
        while (w[earliest] * (1.0 + kSplitTieTolerance) < best)
            ++earliest;
        ties += std::count_if(w.begin(), w.end(), [&](double x) {
                    return x * (1.0 + kSplitTieTolerance) >= best;
                }) > 1;
        CHECK(split_choice(t.red, local) == earliest);
    }
    CHECK(ties == 54);  // the fixture has ties to decide
}

TEST_CASE("Edges and faces are those of the convex reference element", "[element]") {
    // Independent of the descriptors: a face is a maximal set of >= 3 coplanar
    // vertices with every other vertex strictly on one side; an edge is a pair
    // on two faces (3D) or with every other vertex on one side (2D).
    for (Shape s : kShapes) {
        const auto& t = element_type(s);
        const auto x = reference(s);
        const auto n = static_cast<int>(x.size());
        INFO("shape " << static_cast<int>(s));
        std::set<std::set<int>> edges, faces;
        if (t.dim == 2) {
            for (int a = 0; a < n; ++a)
                for (int b = a + 1; b < n; ++b) {
                    int pos = 0, neg = 0;
                    for (int c = 0; c < n; ++c) {
                        if (c == a || c == b)
                            continue;
                        const double side =
                            detail::cross(detail::sub(x[static_cast<std::size_t>(b)],
                                                      x[static_cast<std::size_t>(a)]),
                                          detail::sub(x[static_cast<std::size_t>(c)],
                                                      x[static_cast<std::size_t>(a)]))[2];
                        pos += side > 1e-12;
                        neg += side < -1e-12;
                    }
                    if (pos == 0 || neg == 0)
                        edges.insert({a, b});
                }
        } else {
            for (int a = 0; a < n; ++a)
                for (int b = a + 1; b < n; ++b)
                    for (int c = b + 1; c < n; ++c) {
                        const auto normal =
                            detail::cross(detail::sub(x[static_cast<std::size_t>(b)],
                                                      x[static_cast<std::size_t>(a)]),
                                          detail::sub(x[static_cast<std::size_t>(c)],
                                                      x[static_cast<std::size_t>(a)]));
                        if (detail::dot(normal, normal) < 1e-12)
                            continue;
                        std::set<int> on{a, b, c};
                        int pos = 0, neg = 0;
                        for (int d = 0; d < n; ++d) {
                            if (on.contains(d))
                                continue;
                            const double side =
                                detail::dot(normal,
                                            detail::sub(x[static_cast<std::size_t>(d)],
                                                        x[static_cast<std::size_t>(a)]));
                            if (std::abs(side) < 1e-12)
                                on.insert(d);
                            else
                                (side > 0 ? pos : neg) += 1;
                        }
                        if (pos == 0 || neg == 0)
                            faces.insert(on);
                    }
            for (const auto& f : faces)
                for (int a : f)
                    for (int b : f)
                        if (a < b && std::count_if(faces.begin(), faces.end(), [&](const auto& g) {
                                         return g.contains(a) && g.contains(b);
                                     }) == 2)
                            edges.insert({a, b});
            CHECK(n - static_cast<int>(edges.size()) + static_cast<int>(faces.size()) ==
                  2);  // Euler
        }
        std::set<std::set<int>> declared_edges, declared_faces;
        for (const auto& [a, b] : t.edges)
            declared_edges.insert({a, b});
        for (const auto& f : t.faces)
            declared_faces.insert(std::set<int>(f.begin(), f.end()));
        CHECK(declared_edges == edges);
        CHECK(declared_faces == faces);
    }
}

TEST_CASE("Measures are the divergence theorem over the outward faces", "[element]") {
    // |V| = 1/3 of the flux of x through the boundary, with triangular faces
    // flat and quadrilateral ones bilinear. Independent of signed_measure's
    // formulas, exact on warped elements, and wrong for any face that is not
    // listed cyclically and outward.
    std::mt19937 rng(19);
    std::uniform_real_distribution<double> warp(-0.15, 0.15);
    const std::array<double, 2> g{0.5 - 0.5 / std::sqrt(3.0), 0.5 + 0.5 / std::sqrt(3.0)};
    using detail::cross, detail::dot, detail::sub;
    for (Shape s : kShapes) {
        const auto& t = element_type(s);
        if (t.dim != 3)
            continue;
        INFO("shape " << static_cast<int>(s));
        for (int trial = 0; trial < 50; ++trial) {
            const auto f = random_affine(3, rng);
            std::vector<Point> x;
            for (auto p : reference(s)) {
                for (auto& c : p)
                    c += warp(rng);
                x.push_back(f(p));
            }
            double flux = 0;
            for (const auto& face : t.faces) {
                const auto& a = x[static_cast<std::size_t>(face[0])];
                const auto& b = x[static_cast<std::size_t>(face[1])];
                const auto& c = x[static_cast<std::size_t>(face[2])];
                if (face.size() == 3) {
                    const Point centroid{(a[0] + b[0] + c[0]) / 3,
                                         (a[1] + b[1] + c[1]) / 3,
                                         (a[2] + b[2] + c[2]) / 3};
                    flux += dot(centroid, cross(sub(b, a), sub(c, a))) / 2;
                    continue;
                }
                // x . (x_u x x_v) is biquadratic on a bilinear face: 2x2 Gauss is exact.
                const auto& d = x[static_cast<std::size_t>(face[3])];
                for (double u : g)
                    for (double v : g) {
                        Point p, xu, xv;
                        for (std::size_t k = 0; k < 3; ++k) {
                            p[k] = a[k] * (1 - u) * (1 - v) + b[k] * u * (1 - v) + c[k] * u * v +
                                   d[k] * (1 - u) * v;
                            xu[k] = (b[k] - a[k]) * (1 - v) + (c[k] - d[k]) * v;
                            xv[k] = (d[k] - a[k]) * (1 - u) + (c[k] - b[k]) * u;
                        }
                        flux += dot(p, cross(xu, xv)) / 4;
                    }
            }
            CHECK_THAT(signed_measure(s, x), WithinRel(flux / 3, 1e-12));
        }
    }
}

TEST_CASE("Unknown shapes and malformed input are rejected", "[element]") {
    for (int code = 0; code < 256; ++code)
        CHECK(
            known_shape(static_cast<std::uint8_t>(code)) ==
            (std::find(kShapes.begin(), kShapes.end(), static_cast<Shape>(code)) != kShapes.end()));
    CHECK_THROWS_AS(element_type(static_cast<Shape>(2)), std::invalid_argument);
    CHECK_THROWS_AS(detail::freudenthal(3, std::vector<int>{0, 1, 2}), std::invalid_argument);
    CHECK_THROWS_AS(detail::freudenthal(2, std::vector<int>{0, 0, 1}), std::invalid_argument);
    CHECK_THROWS_AS(detail::freudenthal(3, std::vector<int>{0, 1, 2, 5}), std::invalid_argument);
    CHECK_THROWS_AS(detail::freudenthal(0, std::vector<int>{0}), std::invalid_argument);
    const auto h8 = reference(Shape::H8);
    CHECK_THROWS_AS(signed_measure(Shape::T4, h8), std::invalid_argument);
    CHECK_THROWS_AS(simplex_quality(Shape::H8, h8), std::invalid_argument);
    const auto& t4 = element_type(Shape::T4);
    CHECK_THROWS_AS(split_choice(t4.red, reference(Shape::T4)), std::invalid_argument);
    CHECK(RedTemplate{}.children() == 0);
}
