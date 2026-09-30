/**
 * @file test_util.hpp
 * @brief Mesh builders and independent oracles shared by the engine's test files.
 */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "simplex.hpp"

namespace amr::simplex::test {

using detail::sub;

inline Point midpoint(const Point& a, const Point& b) {
    return {0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.5 * (a[2] + b[2])};
}

inline std::vector<Point> positions(const Mesh& m, Index e) {
    std::vector<Point> x;
    for (Index v : m.element(e))
        x.push_back(m.pos[static_cast<std::size_t>(v)]);
    return x;
}

/// Signed measure of element e (element.hpp's, not adapt's measure()).
inline double signed_measure(const Mesh& m, Index e) {
    return amr::simplex::signed_measure(m.type[static_cast<std::size_t>(e)], positions(m, e));
}

inline double total_measure(const Mesh& m) {
    double sum = 0;
    for (Index e = 0; e < m.num_elements(); ++e)
        sum += signed_measure(m, e);
    return sum;
}

/// Adds an element of shape s, reordering its vertices if needed so that it
/// is positively oriented.
inline void add_oriented(Mesh& m, Shape s, std::vector<Index> v) {
    m.add(s, v);
    if (signed_measure(m, m.num_elements() - 1) > 0)
        return;
    switch (s) {
        case Shape::T3:
        case Shape::T4:
            std::swap(v[0], v[1]);
            break;
        case Shape::Q4:
            v = {v[0], v[3], v[2], v[1]};
            break;
        case Shape::P5:
            v = {v[0], v[3], v[2], v[1], v[4]};
            break;
        case Shape::H8:
            std::rotate(v.begin(), v.begin() + 4, v.end());  // swap bottom and top
            break;
        case Shape::W6:
            std::rotate(v.begin(), v.begin() + 3, v.end());
            break;
    }
    const auto first = m.offset[m.offset.size() - 2];
    std::copy(v.begin(), v.end(), m.con.begin() + static_cast<std::ptrdiff_t>(first));
}

/// Positively oriented simplex of the mesh's dimension.
inline void add_element(Mesh& m, std::vector<Index> v) {
    add_oriented(m, m.dim == 2 ? Shape::T3 : Shape::T4, std::move(v));
}

/// n^dim grid of unit cells, 2 triangles or 6 (Kuhn) tetrahedra per cell,
/// interior nodes jittered so no two diagonals tie.
inline Mesh grid(int dim, int n, double jitter, std::mt19937& rng) {
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

/// Nodes on an integer lattice, created on first use.
struct LatticeMesh {
    Mesh mesh;
    std::map<std::array<int, 3>, Index> id;
    Index at(int x, int y, int z) {
        const auto [it, fresh] = id.try_emplace({x, y, z}, static_cast<Index>(mesh.pos.size()));
        if (fresh)
            mesh.pos.push_back({double(x), double(y), double(z)});
        return it->second;
    }
};

/// n x n cells of side 2, each a quadrilateral or two triangles (either
/// diagonal) at random, the first cell a quadrilateral and the second two
/// triangles: a conforming hybrid 2D mesh with integer nodes. The choices
/// use mt19937's raw output, which the standard fixes, so the mesh is the
/// same with every standard library.
inline Mesh hybrid_2d(int n, std::mt19937& rng) {
    LatticeMesh g;
    g.mesh.dim = 2;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            const Index a = g.at(2 * i, 2 * j, 0), b = g.at(2 * i + 2, 2 * j, 0);
            const Index c = g.at(2 * i + 2, 2 * j + 2, 0), d = g.at(2 * i, 2 * j + 2, 0);
            const int cell = i * n + j;
            switch (cell < 2 ? static_cast<std::uint32_t>(cell) : rng() % 3) {
                case 0:
                    add_oriented(g.mesh, Shape::Q4, {a, b, c, d});
                    break;
                case 1:
                    add_oriented(g.mesh, Shape::T3, {a, b, c});
                    add_oriented(g.mesh, Shape::T3, {a, c, d});
                    break;
                default:
                    add_oriented(g.mesh, Shape::T3, {a, b, d});
                    add_oriented(g.mesh, Shape::T3, {b, c, d});
            }
        }
    return g.mesh;
}

/// n x n columns of 4 cells of side 2, every 3D shape conforming to its
/// neighbours: layer 0 has hexahedra or cells of 6 pyramids around their
/// centre, at random; layer 1 pyramid cells whose top pyramid is cut into 2
/// tetrahedra; layer 2 Kuhn cells of 6 tetrahedra; layer 3 pairs of prisms.
/// Triangulated faces all take the diagonal through the cell's low corner.
/// Layer 0's first column is a hexahedron and its second a pyramid cell; the
/// rest use mt19937's raw output, so the mesh is the same with every
/// standard library.
inline Mesh hybrid_3d(int n, std::mt19937& rng) {
    LatticeMesh g;
    g.mesh.dim = 3;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int layer = 0; layer < 4; ++layer) {
                const int x = 2 * i, y = 2 * j, z = 2 * layer;
                const auto c = [&](int dx, int dy, int dz) {
                    return g.at(x + 2 * dx, y + 2 * dy, z + 2 * dz);
                };
                const int column = i * n + j;
                if (layer == 0 && (column < 2 ? column == 0 : rng() % 2 == 0)) {
                    add_oriented(g.mesh,
                                 Shape::H8,
                                 {c(0, 0, 0),
                                  c(1, 0, 0),
                                  c(1, 1, 0),
                                  c(0, 1, 0),
                                  c(0, 0, 1),
                                  c(1, 0, 1),
                                  c(1, 1, 1),
                                  c(0, 1, 1)});
                } else if (layer <= 1) {
                    const Index centre = g.at(x + 1, y + 1, z + 1);
                    const std::vector<std::array<Index, 4>> faces{
                        {c(0, 0, 0), c(1, 0, 0), c(1, 1, 0), c(0, 1, 0)},  // bottom
                        {c(0, 0, 1), c(1, 0, 1), c(1, 1, 1), c(0, 1, 1)},  // top
                        {c(0, 0, 0), c(1, 0, 0), c(1, 0, 1), c(0, 0, 1)},
                        {c(1, 0, 0), c(1, 1, 0), c(1, 1, 1), c(1, 0, 1)},
                        {c(1, 1, 0), c(0, 1, 0), c(0, 1, 1), c(1, 1, 1)},
                        {c(0, 1, 0), c(0, 0, 0), c(0, 0, 1), c(0, 1, 1)}};
                    for (std::size_t f = 0; f < faces.size(); ++f) {
                        const auto& q = faces[f];
                        if (layer == 1 && f == 1) {  // the top, towards the Kuhn layer
                            add_oriented(g.mesh, Shape::T4, {q[0], q[1], q[2], centre});
                            add_oriented(g.mesh, Shape::T4, {q[0], q[2], q[3], centre});
                        } else {
                            add_oriented(g.mesh, Shape::P5, {q[0], q[1], q[2], q[3], centre});
                        }
                    }
                } else if (layer == 2) {
                    std::array<int, 3> axes{0, 1, 2};
                    do {
                        std::array<int, 3> d{0, 0, 0};
                        std::vector<Index> v{c(0, 0, 0)};
                        for (int a : axes) {
                            ++d[static_cast<std::size_t>(a)];
                            v.push_back(c(d[0], d[1], d[2]));
                        }
                        add_oriented(g.mesh, Shape::T4, v);
                    } while (std::next_permutation(axes.begin(), axes.end()));
                } else {
                    add_oriented(
                        g.mesh,
                        Shape::W6,
                        {c(0, 0, 0), c(1, 0, 0), c(1, 1, 0), c(0, 0, 1), c(1, 0, 1), c(1, 1, 1)});
                    add_oriented(
                        g.mesh,
                        Shape::W6,
                        {c(0, 0, 0), c(1, 1, 0), c(0, 1, 0), c(0, 0, 1), c(1, 1, 1), c(0, 1, 1)});
                }
            }
    return g.mesh;
}

/// A random ceil(fraction * n) of 0..n-1. Fisher-Yates on mt19937's raw
/// output (std::shuffle differs between standard libraries).
inline std::vector<Index> random_subset(Index n, double fraction, std::mt19937& rng) {
    std::vector<Index> all(static_cast<std::size_t>(n));
    std::iota(all.begin(), all.end(), 0);
    for (std::size_t i = all.size(); i > 1; --i)
        std::swap(all[i - 1], all[rng() % i]);
    all.resize(static_cast<std::size_t>(std::ceil(fraction * n)));
    return all;
}

/// The MATLAB loop, verbatim but for k parents: sweep the hanging nodes until
/// nothing changes.
inline std::vector<Index> naive_closure(const Mesh& m, const std::vector<Index>& list) {
    std::vector<char> sel(static_cast<std::size_t>(m.num_elements()), 0);
    for (Index e : list)
        sel[static_cast<std::size_t>(e)] = 1;
    const auto has = [&m](Index e, Index v) { return detail::contains(m.element(e), v); };
    for (bool changed = true; changed;) {
        changed = false;
        for (const auto& h : m.hn) {
            bool touched = false;
            for (Index e = 0; e < m.num_elements(); ++e)
                touched |= sel[static_cast<std::size_t>(e)] && has(e, h.node);
            if (!touched)
                continue;
            for (Index e = 0; e < m.num_elements(); ++e)
                if (!sel[static_cast<std::size_t>(e)] && !has(e, h.node) &&
                    std::all_of(
                        h.parent.begin(), h.parent.end(), [&](Index p) { return has(e, p); })) {
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
inline Mesh randomly_refined(Mesh m, int cycles, double fraction, std::mt19937& rng) {
    for (int c = 0; c < cycles; ++c)
        m = refine(m, balance_closure(m, random_subset(m.num_elements(), fraction, rng))).mesh;
    return m;
}

inline double tet_quality(const Point& a, const Point& b, const Point& c, const Point& d) {
    const std::array<Point, 4> x{a, b, c, d};
    return simplex_quality(Shape::T4, x);
}

/// Worst child quality of the best octahedron split, enumerating the three
/// diagonals (midpoints of opposite edges) and the equator around each.
inline double best_octahedron_worst_quality(const std::array<Point, 4>& c) {
    std::map<std::pair<int, int>, Point> mid;
    for (int a = 0; a < 4; ++a)
        for (int b = a + 1; b < 4; ++b)
            mid[{a, b}] = midpoint(c[static_cast<std::size_t>(a)], c[static_cast<std::size_t>(b)]);
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
            worst = std::min(worst,
                             tet_quality(mid[d0], mid[d1], mid[ring[i]], mid[ring[(i + 1) % 4]]));
        best = std::max(best, worst);
    }
    return best;
}

/**
 * @brief Everything that makes a mesh with hanging nodes conforming, checked
 * from the geometry alone; empty if it holds.
 *
 * For meshes of the box [lo, hi] with dyadic coordinates (so that every sum
 * below is exact) and planar faces:
 * - Continuity: a random integer field on the free nodes, extended to the
 *   hanging ones by their constraints, has the same trace from every element
 *   at every point of a 4 x 4 lattice on each face (edges in 2D; triangular
 *   faces linear, quadrilateral ones bilinear). A missing, wrong or misplaced
 *   constraint, or two neighbours refining a face differently, breaks it.
 * - Tiling: every element is positively oriented, and every one of
 *   `samples` random points of the box lies in exactly one element (none
 *   when the mesh does not fill a box).
 * - Hanging set: a node is constrained exactly when it lies on the boundary
 *   of an element that does not have it as a vertex, and no constraint's
 *   parent is constrained itself (1-irregular). Constraints sit at the
 *   weighted mean of their parents.
 * - No two nodes share a position.
 */
inline std::string conformity_errors(
    const Mesh& m, const Point& lo, const Point& hi, int samples, std::mt19937& rng) {
    std::ostringstream err;
    const auto n = m.pos.size();
    const auto P = [&m](Index v) { return m.pos[static_cast<std::size_t>(v)]; };

    // Constraint positions, 1-irregularity and distinct nodes.
    std::vector<char> constrained(n, 0);
    for (const auto& h : m.hn)
        constrained[static_cast<std::size_t>(h.node)] = 1;
    for (const auto& h : m.hn) {
        Point p{0, 0, 0};
        for (std::size_t k = 0; k < h.parent.size(); ++k) {
            if (constrained[static_cast<std::size_t>(h.parent[k])])
                err << "node " << h.node << " hangs on hanging node " << h.parent[k] << "\n";
            for (std::size_t c = 0; c < 3; ++c)
                p[c] += h.weight[k] * P(h.parent[k])[c];
        }
        if (p != P(h.node))
            err << "node " << h.node << " is not at its parents' weighted mean\n";
    }
    if (std::set<Point>(m.pos.begin(), m.pos.end()).size() != n)
        err << "two nodes share a position\n";

    // The field: random integers on free nodes, constraints on hanging ones.
    std::uniform_int_distribution<int> pick(-1000, 1000);
    std::vector<double> u(n);
    for (auto& x : u)
        x = pick(rng);
    for (const auto& h : m.hn) {
        double v = 0;
        for (std::size_t k = 0; k < h.parent.size(); ++k)
            v += h.weight[k] * u[static_cast<std::size_t>(h.parent[k])];
        u[static_cast<std::size_t>(h.node)] = v;
    }

    // Face traces, keyed by exact position.
    constexpr int q = 4;
    std::map<Point, std::pair<double, std::set<Index>>> trace;
    const auto sample = [&](Index e, const std::vector<Index>& f, std::vector<double> w) {
        Point p{0, 0, 0};
        double v = 0;
        for (std::size_t k = 0; k < f.size(); ++k) {
            for (std::size_t c = 0; c < 3; ++c)
                p[c] += w[k] * P(f[k])[c];
            v += w[k] * u[static_cast<std::size_t>(f[k])];
        }
        const auto [it, fresh] = trace.try_emplace(p, v, std::set<Index>{});
        if (!fresh && it->second.first != v)
            err << "field jumps at (" << p[0] << ", " << p[1] << ", " << p[2] << ")\n";
        it->second.second.insert(e);
    };
    for (Index e = 0; e < m.num_elements(); ++e) {
        const auto& t = element_type(m.type[static_cast<std::size_t>(e)]);
        const auto v = m.element(e);
        std::vector<std::vector<Index>> faces;
        if (t.dim == 2)
            for (const auto& [a, b] : t.edges)
                faces.push_back({v[static_cast<std::size_t>(a)], v[static_cast<std::size_t>(b)]});
        else
            for (const auto& face : t.faces) {
                faces.emplace_back();
                for (int k : face)
                    faces.back().push_back(v[static_cast<std::size_t>(k)]);
            }
        for (const auto& f : faces)
            for (int i = 0; i <= q; ++i)
                for (int j = 0; j <= q; ++j) {
                    const double s = double(i) / q, r = double(j) / q;
                    if (f.size() == 2 && j == 0)
                        sample(e, f, {1 - s, s});
                    else if (f.size() == 3 && i + j <= q)
                        sample(e, f, {1 - s - r, s, r});
                    else if (f.size() == 4)
                        sample(e, f, {(1 - s) * (1 - r), s * (1 - r), s * r, (1 - s) * r});
                }
    }
    for (Index v = 0; v < static_cast<Index>(n); ++v) {
        const auto it = trace.find(P(v));
        bool on_foreign = false;
        if (it != trace.end())
            for (Index e : it->second.second)
                on_foreign |= !detail::contains(m.element(e), v);
        if (on_foreign != static_cast<bool>(constrained[static_cast<std::size_t>(v)]))
            err << "node " << v
                << (on_foreign ? " hangs but is not constrained\n"
                               : " is constrained but does not hang\n");
    }

    // Tiling: orientation, and each random point in exactly one element.
    std::vector<std::vector<std::pair<Point, Point>>> halfspaces;  // (a, outward normal)
    for (Index e = 0; e < m.num_elements(); ++e) {
        if (!(signed_measure(m, e) > 0))
            err << "element " << e << " is not positively oriented\n";
        const auto& t = element_type(m.type[static_cast<std::size_t>(e)]);
        const auto x = positions(m, e);
        std::vector<std::pair<Point, Point>> hs;
        if (t.dim == 2) {
            for (std::size_t k = 0; k < x.size(); ++k) {  // counter-clockwise polygon
                const Point d = sub(x[(k + 1) % x.size()], x[k]);
                hs.push_back({x[k], {d[1], -d[0], 0}});
            }
        } else {
            for (const auto& face : t.faces) {
                const auto& a = x[static_cast<std::size_t>(face[0])];
                hs.push_back({a,
                              detail::cross(sub(x[static_cast<std::size_t>(face[1])], a),
                                            sub(x[static_cast<std::size_t>(face[2])], a))});
            }
        }
        halfspaces.push_back(std::move(hs));
    }
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    for (int s = 0; s < samples; ++s) {
        Point p{};
        for (std::size_t c = 0; c < static_cast<std::size_t>(m.dim); ++c)
            p[c] = lo[c] + unit(rng) * (hi[c] - lo[c]);
        int inside = 0;
        for (const auto& hs : halfspaces)
            inside += std::all_of(hs.begin(), hs.end(), [&p](const auto& h) {
                return detail::dot(h.second, sub(p, h.first)) < 0;
            });
        if (inside != 1)
            err << "point (" << p[0] << ", " << p[1] << ", " << p[2] << ") is in " << inside
                << " elements\n";
    }
    return err.str();
}

}  // namespace amr::simplex::test
