/**
 * @file element.hpp
 * @brief Reference elements and their red refinement, from one model.
 *
 * @details Every element here except the pyramid is a product of simplices,
 * Δ^{d1} × ... × Δ^{dm}: a triangle is Δ², a tetrahedron Δ³, a quadrilateral
 * Δ¹×Δ¹, a hexahedron Δ¹×Δ¹×Δ¹ and a prism Δ²×Δ¹. Its red refinement is the
 * product of its factors' Freudenthal subdivisions [Freudenthal1942, Bey2000]:
 * a d-simplex splits into 2^d simplices whose vertices are the points with
 * barycentric coordinates in {0, 1/2, 1}, and a product splits into the
 * products of its factors' children, 2^(d1 + ... + dm) children of its own
 * shape. A new node is such a lattice point, a weighted mean of the parent's
 * vertices whose weights are the parent's shape functions there. So one row
 * gives the node's coordinates, its prolongation row and, if it hangs, its
 * constraint.
 *
 * A d-simplex with d >= 3 has several Freudenthal subdivisions, one per vertex
 * ordering (a tetrahedron's 3 octahedron diagonals). split_choice() picks the
 * one whose worst child has the highest mean-ratio quality [Zhang1995]; ties
 * go to the earliest candidate.
 *
 * The pyramid is not a product of simplices. Its red refinement, 6 pyramids
 * (at the base corners, the apex and the centre, inverted) and 4 tetrahedra,
 * is given by hand.
 *
 * T3 and T4 keep the child and new-node order the FE-DIC solver's meshes have
 * always had (mesh_subdivide_list, observed from its output), since
 * correlation round-off depends on numbering; the tests check that these
 * orders list exactly the generated subdivisions. Codes follow Correli's
 * element numbering where it has the element (T3 = 0, Q4 = 1, T4 = 4, H8 = 5,
 * W6 = 8) and VTK's for the pyramid (P5 = 14).
 */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace amr::simplex {

using Point = std::array<double, 3>;

enum class Shape : std::uint8_t { T3 = 0, Q4 = 1, T4 = 4, H8 = 5, W6 = 8, P5 = 14 };

/// A new node of a refined element: a weighted mean of the parent's vertices.
struct LatticeNode {
    std::vector<int> vertex;     ///< parent vertices (local), ascending
    std::vector<double> weight;  ///< their weights, summing to 1
};

/// A child: its shape and local nodes (0..nv-1 the parent's vertices, nv + k
/// the parent's new node k).
struct Child {
    Shape shape;
    std::vector<int> node;
};

/// Red refinement of a reference element. The children are head, then one of
/// the choices, then tail; most shapes have a single (empty) choice.
struct RedTemplate {
    std::vector<LatticeNode> new_nodes;
    std::vector<Child> head, tail;
    std::vector<std::vector<Child>> choices;
    [[nodiscard]] std::size_t children() const {
        return head.size() + choices.front().size() + tail.size();
    }
};

struct ElementType {
    Shape shape = Shape::T3;
    int dim = 2;       ///< topological dimension
    int vertices = 3;  ///< in the element's node order
    std::vector<std::array<int, 2>> edges;
    std::vector<std::vector<int>> faces;  ///< 3D: codimension-1 faces as ascending vertex lists
    RedTemplate red;
};

namespace detail {

/// Freudenthal subdivision of the d-simplex into 2^d children, with the
/// parent's vertices taken in `order` (a permutation of 0..d). Points are
/// barycentric coordinates times 2; each child lists its points in Kuhn path
/// order, then, if `oriented`, with its first two swapped when that order is
/// negative, so every child has the parent's orientation.
struct SimplexSplit {
    std::vector<std::vector<int>> point;  ///< barycentric * 2, one entry per vertex
    std::vector<std::vector<int>> child;  ///< indices into point
};

inline std::vector<std::vector<int>> lattice_points(int d) {
    std::vector<std::vector<int>> points;
    std::vector<int> b(static_cast<std::size_t>(d + 1), 0);
    for (int i = 0; i <= d; ++i) {  // vertices first, then edge midpoints
        std::fill(b.begin(), b.end(), 0);
        b[static_cast<std::size_t>(i)] = 2;
        points.push_back(b);
    }
    for (int i = 0; i <= d; ++i)
        for (int j = i + 1; j <= d; ++j) {
            std::fill(b.begin(), b.end(), 0);
            b[static_cast<std::size_t>(i)] = b[static_cast<std::size_t>(j)] = 1;
            points.push_back(b);
        }
    return points;
}

/// Sign of the determinant of an integer matrix (Bareiss, exact).
inline int det_sign(std::vector<std::vector<long long>> a) {
    const std::size_t n = a.size();
    int sign = 1;
    long long prev = 1;
    for (std::size_t k = 0; k + 1 < n; ++k) {
        if (a[k][k] == 0) {
            std::size_t r = k + 1;
            while (r < n && a[r][k] == 0)
                ++r;
            if (r == n)
                return 0;
            std::swap(a[k], a[r]);
            sign = -sign;
        }
        for (std::size_t i = k + 1; i < n; ++i)
            for (std::size_t j = k + 1; j < n; ++j)
                a[i][j] = (a[i][j] * a[k][k] - a[i][k] * a[k][j]) / prev;
        prev = a[k][k];
    }
    const long long last = a[n - 1][n - 1];
    return last == 0 ? 0 : (last > 0 ? sign : -sign);
}

inline SimplexSplit freudenthal(int d, std::span<const int> order, bool oriented = true) {
    SimplexSplit s;
    s.point = lattice_points(d);
    std::map<std::vector<int>, int> index;
    for (std::size_t i = 0; i < s.point.size(); ++i)
        index[s.point[i]] = static_cast<int>(i);
    // Kuhn coordinates x with 2 >= x1 >= ... >= xd >= 0; in the ordered frame
    // lambda'_0 = 2 - x1, lambda'_i = x_i - x_{i+1}, lambda'_d = x_d.
    const auto barycentric = [&](const std::vector<int>& x) {
        std::vector<int> lambda(static_cast<std::size_t>(d + 1));
        for (int i = 0; i <= d; ++i) {
            const int hi = i == 0 ? 2 : x[static_cast<std::size_t>(i - 1)];
            const int lo = i == d ? 0 : x[static_cast<std::size_t>(i)];
            lambda[static_cast<std::size_t>(order[static_cast<std::size_t>(i)])] = hi - lo;
        }
        return lambda;
    };
    const auto inside = [d](const std::vector<int>& x) {
        for (int i = 0; i < d; ++i) {
            const int hi = i == 0 ? 2 : x[static_cast<std::size_t>(i - 1)];
            if (x[static_cast<std::size_t>(i)] > hi || x[static_cast<std::size_t>(i)] < 0)
                return false;
        }
        return true;
    };
    std::vector<int> perm(static_cast<std::size_t>(d));
    for (int z = 0; z < (1 << d); ++z) {
        std::iota(perm.begin(), perm.end(), 0);
        do {
            std::vector<int> x(static_cast<std::size_t>(d));
            for (int i = 0; i < d; ++i)
                x[static_cast<std::size_t>(i)] = (z >> i) & 1;
            std::vector<std::vector<int>> path{x};
            bool ok = inside(x);
            for (int k = 0; k < d && ok; ++k) {
                ++x[static_cast<std::size_t>(perm[static_cast<std::size_t>(k)])];
                ok = inside(x);
                path.push_back(x);
            }
            if (!ok)
                continue;
            std::vector<int> child;
            std::vector<std::vector<long long>> b;
            for (const auto& p : path) {
                const auto lambda = barycentric(p);
                child.push_back(index.at(lambda));
                b.emplace_back(lambda.begin(), lambda.end());
            }
            if (oriented && det_sign(b) < 0)
                std::swap(child[0], child[1]);
            s.child.push_back(child);
        } while (std::next_permutation(perm.begin(), perm.end()));
    }
    return s;
}

/// The distinct Freudenthal subdivisions of the d-simplex over all vertex
/// orderings, in order of first appearance (the identity ordering first).
inline std::vector<SimplexSplit> freudenthal_candidates(int d) {
    std::vector<SimplexSplit> out;
    std::set<std::set<std::set<int>>> seen;
    std::vector<int> order(static_cast<std::size_t>(d + 1));
    std::iota(order.begin(), order.end(), 0);
    do {
        auto s = freudenthal(d, order);
        std::set<std::set<int>> key;
        for (const auto& c : s.child)
            key.insert(std::set<int>(c.begin(), c.end()));
        if (seen.insert(key).second)
            out.push_back(std::move(s));
    } while (std::next_permutation(order.begin(), order.end()));
    return out;
}

/// Red refinement of a product of simplices (factor dimensions `factors`)
/// whose vertex k in element order is the product vertex lex_of[k]; product
/// vertices are numbered lexicographically, the first factor slowest.
inline RedTemplate product_red(Shape shape,
                               const std::vector<int>& factors,
                               const std::vector<int>& lex_of) {
    const std::size_t m = factors.size();
    std::vector<SimplexSplit> split;
    for (int d : factors) {
        std::vector<int> identity(static_cast<std::size_t>(d + 1));
        std::iota(identity.begin(), identity.end(), 0);
        split.push_back(freudenthal(d, identity));
    }
    std::vector<int> element_of_lex(lex_of.size());
    for (std::size_t k = 0; k < lex_of.size(); ++k)
        element_of_lex[static_cast<std::size_t>(lex_of[k])] = static_cast<int>(k);
    // Product lattice points, lexicographic over the factors' point lists.
    std::vector<std::vector<int>> tuples{{}};
    for (std::size_t j = 0; j < m; ++j) {
        std::vector<std::vector<int>> next;
        for (const auto& t : tuples)
            for (std::size_t p = 0; p < split[j].point.size(); ++p) {
                auto u = t;
                u.push_back(static_cast<int>(p));
                next.push_back(u);
            }
        tuples = std::move(next);
    }
    const auto lex_vertex = [&](const std::vector<int>& factor_vertex) {
        int idx = 0;
        for (std::size_t j = 0; j < m; ++j)
            idx = idx * (factors[j] + 1) + factor_vertex[j];
        return idx;
    };
    RedTemplate red;
    std::map<std::vector<int>, int> local;  // lattice tuple -> local node
    for (const auto& t : tuples) {
        // Weight of each product vertex: the product of the factor weights.
        std::map<int, double> w;
        std::vector<int> fv(m, 0);
        std::function<void(std::size_t, double)> expand = [&](std::size_t j, double acc) {
            if (j == m) {
                w[element_of_lex[static_cast<std::size_t>(lex_vertex(fv))]] += acc;
                return;
            }
            const auto& b = split[j].point[static_cast<std::size_t>(t[j])];
            for (std::size_t i = 0; i < b.size(); ++i)
                if (b[i] != 0) {
                    fv[j] = static_cast<int>(i);
                    expand(j + 1, acc * b[i] / 2.0);
                }
        };
        expand(0, 1.0);
        if (w.size() == 1) {
            local[t] = w.begin()->first;  // a parent vertex
            continue;
        }
        LatticeNode node;
        for (const auto& [v, x] : w) {
            node.vertex.push_back(v);
            node.weight.push_back(x);
        }
        local[t] = static_cast<int>(lex_of.size() + red.new_nodes.size());
        red.new_nodes.push_back(std::move(node));
    }
    // Children: products of the factors' children, in element vertex order.
    std::vector<std::vector<int>> combos{{}};
    for (std::size_t j = 0; j < m; ++j) {
        std::vector<std::vector<int>> next;
        for (const auto& c : combos)
            for (std::size_t k = 0; k < split[j].child.size(); ++k) {
                auto u = c;
                u.push_back(static_cast<int>(k));
                next.push_back(u);
            }
        combos = std::move(next);
    }
    for (const auto& combo : combos) {
        std::vector<int> lex_nodes;  // the child's vertices, lexicographic
        std::vector<std::vector<int>> t{{}};
        for (std::size_t j = 0; j < m; ++j) {
            std::vector<std::vector<int>> next;
            for (const auto& u : t)
                for (int p : split[j].child[static_cast<std::size_t>(combo[j])]) {
                    auto v = u;
                    v.push_back(p);
                    next.push_back(v);
                }
            t = std::move(next);
        }
        for (const auto& u : t)
            lex_nodes.push_back(local.at(u));
        Child child{shape, {}};
        for (int k : lex_of)
            child.node.push_back(lex_nodes[static_cast<std::size_t>(k)]);
        red.head.push_back(std::move(child));
    }
    red.choices = {{}};
    return red;
}

inline LatticeNode midpoint_node(int a, int b) {
    return {{std::min(a, b), std::max(a, b)}, {0.5, 0.5}};
}

inline ElementType make_t3() {
    ElementType t{Shape::T3, 2, 3, {{{0, 1}}, {{1, 2}}, {{0, 2}}}, {}, {}};
    // New nodes 3 = m12, 4 = m02, 5 = m01, created in this order.
    t.red.new_nodes = {midpoint_node(1, 2), midpoint_node(0, 2), midpoint_node(0, 1)};
    for (const auto& c : std::vector<std::vector<int>>{{2, 4, 3}, {3, 5, 1}, {4, 0, 5}, {5, 3, 4}})
        t.red.head.push_back({Shape::T3, c});
    t.red.choices = {{}};
    return t;
}

inline ElementType make_t4() {
    ElementType t{Shape::T4,
                  3,
                  4,
                  {{{0, 1}}, {{0, 2}}, {{0, 3}}, {{1, 2}}, {{1, 3}}, {{2, 3}}},
                  {{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}},
                  {}};
    // New nodes 4 = m13, 5 = m03, 6 = m01, 7 = m23, 8 = m12, 9 = m02.
    t.red.new_nodes = {midpoint_node(1, 3),
                       midpoint_node(0, 3),
                       midpoint_node(0, 1),
                       midpoint_node(2, 3),
                       midpoint_node(1, 2),
                       midpoint_node(0, 2)};
    t.red.head = {{Shape::T4, {3, 7, 4, 5}}};
    // The octahedron's 4 children for each diagonal: m13-m02, m01-m23, m03-m12.
    const std::vector<std::vector<std::vector<int>>> octahedron{
        {{4, 6, 5, 9}, {4, 8, 6, 9}, {4, 7, 8, 9}, {4, 5, 7, 9}},
        {{6, 7, 5, 4}, {6, 7, 9, 5}, {6, 7, 8, 9}, {6, 7, 4, 8}},
        {{5, 8, 4, 6}, {5, 8, 6, 9}, {5, 8, 9, 7}, {5, 8, 7, 4}}};
    for (const auto& split : octahedron) {
        std::vector<Child> choice;
        for (const auto& c : split)
            choice.push_back({Shape::T4, c});
        t.red.choices.push_back(std::move(choice));
    }
    t.red.tail = {{Shape::T4, {4, 8, 1, 6}}, {Shape::T4, {5, 9, 6, 0}}, {Shape::T4, {7, 2, 8, 9}}};
    return t;
}

inline ElementType make_q4() {
    // Counter-clockwise: (0,0), (1,0), (1,1), (0,1); lexicographic (x slowest).
    ElementType t{Shape::Q4, 2, 4, {{{0, 1}}, {{1, 2}}, {{2, 3}}, {{0, 3}}}, {}, {}};
    t.red = product_red(Shape::Q4, {1, 1}, {0, 2, 3, 1});
    return t;
}

inline ElementType make_h8() {
    // Bottom face counter-clockwise from (0,0,0), then the top face above it.
    ElementType t{
        Shape::H8,
        3,
        8,
        {{{0, 1}},
         {{1, 2}},
         {{2, 3}},
         {{0, 3}},
         {{4, 5}},
         {{5, 6}},
         {{6, 7}},
         {{4, 7}},
         {{0, 4}},
         {{1, 5}},
         {{2, 6}},
         {{3, 7}}},
        {{0, 1, 2, 3}, {4, 5, 6, 7}, {0, 1, 4, 5}, {1, 2, 5, 6}, {2, 3, 6, 7}, {0, 3, 4, 7}},
        {}};
    t.red = product_red(Shape::H8, {1, 1, 1}, {0, 4, 6, 2, 1, 5, 7, 3});
    return t;
}

inline ElementType make_w6() {
    // Bottom triangle, then the top triangle above it; lexicographic
    // (triangle vertex slowest, then the segment).
    ElementType t{
        Shape::W6,
        3,
        6,
        {{{0, 1}}, {{1, 2}}, {{0, 2}}, {{3, 4}}, {{4, 5}}, {{3, 5}}, {{0, 3}}, {{1, 4}}, {{2, 5}}},
        {{0, 1, 2}, {3, 4, 5}, {0, 1, 3, 4}, {1, 2, 4, 5}, {0, 2, 3, 5}},
        {}};
    t.red = product_red(Shape::W6, {2, 1}, {0, 2, 4, 1, 3, 5});
    return t;
}

inline ElementType make_p5() {
    // Base 0-1-2-3 counter-clockwise seen from the apex 4.
    ElementType t{Shape::P5,
                  3,
                  5,
                  {{{0, 1}}, {{1, 2}}, {{2, 3}}, {{0, 3}}, {{0, 4}}, {{1, 4}}, {{2, 4}}, {{3, 4}}},
                  {{0, 1, 2, 3}, {0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {0, 3, 4}},
                  {}};
    // 5 = m01, 6 = m12, 7 = m23, 8 = m03, 9 = base centre, 10..13 = m04..m34.
    t.red.new_nodes = {midpoint_node(0, 1),
                       midpoint_node(1, 2),
                       midpoint_node(2, 3),
                       midpoint_node(0, 3),
                       {{0, 1, 2, 3}, {0.25, 0.25, 0.25, 0.25}},
                       midpoint_node(0, 4),
                       midpoint_node(1, 4),
                       midpoint_node(2, 4),
                       midpoint_node(3, 4)};
    t.red.head = {{Shape::P5, {0, 5, 9, 8, 10}},
                  {Shape::P5, {5, 1, 6, 9, 11}},  // base corners
                  {Shape::P5, {9, 6, 2, 7, 12}},
                  {Shape::P5, {8, 9, 7, 3, 13}},
                  {Shape::P5, {10, 11, 12, 13, 4}},  // apex
                  {Shape::P5, {10, 13, 12, 11, 9}},  // centre, inverted
                  {Shape::T4, {11, 5, 10, 9}},
                  {Shape::T4, {12, 6, 11, 9}},  // one per lateral face
                  {Shape::T4, {13, 7, 12, 9}},
                  {Shape::T4, {10, 8, 13, 9}}};
    t.red.choices = {{}};
    return t;
}

}  // namespace detail

inline bool known_shape(std::uint8_t code) {
    switch (static_cast<Shape>(code)) {
        case Shape::T3:
        case Shape::Q4:
        case Shape::T4:
        case Shape::H8:
        case Shape::W6:
        case Shape::P5:
            return true;
    }
    return false;
}

/// The reference element of a shape, built once.
inline const ElementType& element_type(Shape s) {
    static const std::map<Shape, ElementType> types{{Shape::T3, detail::make_t3()},
                                                    {Shape::Q4, detail::make_q4()},
                                                    {Shape::T4, detail::make_t4()},
                                                    {Shape::H8, detail::make_h8()},
                                                    {Shape::W6, detail::make_w6()},
                                                    {Shape::P5, detail::make_p5()}};
    const auto it = types.find(s);
    if (it == types.end())
        throw std::invalid_argument("element: unknown shape code " +
                                    std::to_string(static_cast<int>(s)));
    return it->second;
}

namespace detail {

inline Point sub(const Point& a, const Point& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
inline double dot(const Point& a, const Point& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
inline Point cross(const Point& a, const Point& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
inline double tet_signed(const Point& a, const Point& b, const Point& c, const Point& d) {
    return dot(cross(sub(b, a), sub(c, a)), sub(d, a)) / 6.0;
}

}  // namespace detail

/// Signed area (2D, in the xy plane) or volume (3D) of an element whose
/// vertices, in its node order, are x.
inline double signed_measure(Shape s, std::span<const Point> x) {
    using namespace detail;
    switch (s) {
        case Shape::T3:
            return 0.5 * cross(sub(x[1], x[0]), sub(x[2], x[0]))[2];
        case Shape::Q4:
            return 0.5 * cross(sub(x[2], x[0]), sub(x[3], x[1]))[2];
        case Shape::T4:
            return tet_signed(x[0], x[1], x[2], x[3]);
        case Shape::P5:
            return tet_signed(x[0], x[1], x[2], x[4]) + tet_signed(x[0], x[2], x[3], x[4]);
        case Shape::H8: {  // 2x2x2 Gauss on the trilinear map: exact
            const double g[2]{0.5 - 0.5 / std::sqrt(3.0), 0.5 + 0.5 / std::sqrt(3.0)};
            constexpr int ref[8][3]{{0, 0, 0},
                                    {1, 0, 0},
                                    {1, 1, 0},
                                    {0, 1, 0},
                                    {0, 0, 1},
                                    {1, 0, 1},
                                    {1, 1, 1},
                                    {0, 1, 1}};
            double v = 0.0;
            for (double a : g)
                for (double b : g)
                    for (double c : g) {
                        const double q[3]{a, b, c};
                        Point j[3]{};
                        for (int k = 0; k < 8; ++k)
                            for (int r = 0; r < 3; ++r) {  // d N_k / d q_r
                                double dn = ref[k][r] ? 1.0 : -1.0;
                                for (int o = 0; o < 3; ++o)
                                    if (o != r)
                                        dn *= ref[k][o] ? q[o] : 1.0 - q[o];
                                for (int c3 = 0; c3 < 3; ++c3)
                                    j[r][static_cast<std::size_t>(c3)] +=
                                        dn * x[static_cast<std::size_t>(k)]
                                              [static_cast<std::size_t>(c3)];
                            }
                        v += dot(cross(j[0], j[1]), j[2]) / 8.0;
                    }
            return v;
        }
        case Shape::W6: {  // centroid x 2-point Gauss in the height: exact
            const double g[2]{0.5 - 0.5 / std::sqrt(3.0), 0.5 + 0.5 / std::sqrt(3.0)};
            double v = 0.0;
            for (double z : g) {
                Point p[3], dz{};
                for (int i = 0; i < 3; ++i) {
                    const auto& lo = x[static_cast<std::size_t>(i)];
                    const auto& hi = x[static_cast<std::size_t>(i + 3)];
                    for (std::size_t c = 0; c < 3; ++c) {
                        p[i][c] = (1 - z) * lo[c] + z * hi[c];
                        dz[c] += (hi[c] - lo[c]) / 3.0;
                    }
                }
                v += 0.25 * dot(cross(sub(p[1], p[0]), sub(p[2], p[0])), dz);
            }
            return v;
        }
    }
    throw std::invalid_argument("element: unknown shape");
}

/// Mean-ratio quality of a simplex: 1 when regular, 0 when flat.
inline double simplex_quality(Shape s, std::span<const Point> x) {
    using namespace detail;
    double sum_l2 = 0.0;
    for (std::size_t a = 0; a < x.size(); ++a)
        for (std::size_t b = a + 1; b < x.size(); ++b)
            sum_l2 += dot(sub(x[b], x[a]), sub(x[b], x[a]));
    sum_l2 = std::max(sum_l2, std::numeric_limits<double>::epsilon());
    const double m = std::abs(signed_measure(s, x));
    if (s == Shape::T3)
        return 4.0 * std::sqrt(3.0) * m / sum_l2;
    return 12.0 * std::pow(3.0 * m, 2.0 / 3.0) / sum_l2;
}

/// The choice of `red` whose worst child is best (earliest on ties), given
/// the positions of the parent's vertices and new nodes, in local order.
inline std::size_t split_choice(const RedTemplate& red, std::span<const Point> local) {
    if (red.choices.size() < 2)
        return 0;
    std::size_t best = 0;
    double best_worst = -1.0;
    for (std::size_t k = 0; k < red.choices.size(); ++k) {
        double worst = std::numeric_limits<double>::infinity();
        for (const auto& c : red.choices[k]) {
            std::vector<Point> x;
            for (int n : c.node)
                x.push_back(local[static_cast<std::size_t>(n)]);
            worst = std::min(worst, simplex_quality(c.shape, x));
        }
        if (worst > best_worst) {
            best_worst = worst;
            best = k;
        }
    }
    return best;
}

}  // namespace amr::simplex
