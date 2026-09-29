/**
 * @file simplex.hpp
 * @brief Red refinement of unstructured triangle (T3) and tetrahedron (T4)
 * meshes with hanging nodes and 1-irregular balance.
 *
 * @details The simplicial counterpart of the Morton octree. It holds the
 * unstructured meshes the FE-DIC solver correlates on and refines listed
 * elements, reproducing the MATLAB `mesh_subdivide_list` and
 * `mesh_balance_refine_list` it replaces up to node and element numbering:
 *
 * - A triangle splits into its 4 edge-midpoint triangles.
 * - A tetrahedron splits into its 4 corner tetrahedra plus the inner
 *   octahedron cut along one of its 3 diagonals [Bey1995]. The diagonal is
 *   the one whose worst child has the highest mean-ratio quality
 *   12 (3|V|)^(2/3) / sum(l^2), which keeps the shape classes of a
 *   structured seed bounded [Zhang1995]. Exact ties go to the first
 *   diagonal in kOctahedron order, which depends on the parent's vertex
 *   order.
 *
 * Midpoints are keyed by their edge, so neighbours share them, and a
 * listed hanging node on that edge is reused rather than duplicated. After
 * refinement a node hangs while some element still has its parent edge.
 * Lists are closed under balance_closure() first, which keeps the mesh
 * 1-irregular: no hanging node has a hanging parent.
 *
 * Children keep the parent's orientation. A new node's coordinates are
 * computed as 0.5 * (a + b), which is exactly the value MATLAB's S * pos
 * produces, so meshes can be compared coordinate-exact.
 */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace amr::simplex {

using Index = std::int32_t;
using Point = std::array<double, 3>;

/// Unstructured T3 (dim 2) or T4 (dim 3) mesh. Indices are 0-based.
struct Mesh {
    int dim = 2;
    std::vector<Point> pos;                ///< z = 0 in 2D, as in the DIC meshes
    std::vector<Index> con;                ///< row-major, dim + 1 vertices per element
    std::vector<std::array<Index, 3>> hn;  ///< {hanging node, parent, parent}

    [[nodiscard]] int vertices_per_element() const { return dim + 1; }
    [[nodiscard]] Index num_elements() const {
        return static_cast<Index>(con.size() / static_cast<std::size_t>(vertices_per_element()));
    }
    [[nodiscard]] std::span<const Index> element(Index e) const {
        const auto nv = static_cast<std::size_t>(vertices_per_element());
        return {con.data() + static_cast<std::size_t>(e) * nv, nv};
    }
};

/// A refined mesh and its prolongation U_new = S U_old: node i < n_old keeps
/// its value, and new node n_old + k takes the mean of parents[k].
struct Refinement {
    Mesh mesh;
    std::vector<std::array<Index, 2>> parents;
};

namespace detail {

inline std::uint64_t edge_key(Index a, Index b) {
    if (a > b)
        std::swap(a, b);
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)) << 32) |
           static_cast<std::uint32_t>(b);
}

/// Items grouped by an integer key, in compressed-row form.
struct Csr {
    std::vector<Index> start;  ///< size n_keys + 1
    std::vector<Index> items;
    [[nodiscard]] std::span<const Index> of(Index key) const {
        const auto b = static_cast<std::size_t>(start[static_cast<std::size_t>(key)]);
        const auto e = static_cast<std::size_t>(start[static_cast<std::size_t>(key) + 1]);
        return {items.data() + b, e - b};
    }
};

/// Groups value_of(i) under key_of(i) for i in [0, n_items).
template <class KeyOf, class ValueOf>
Csr make_csr(std::size_t n_keys, std::size_t n_items, KeyOf key_of, ValueOf value_of) {
    Csr csr;
    csr.start.assign(n_keys + 1, 0);
    for (std::size_t i = 0; i < n_items; ++i)
        ++csr.start[static_cast<std::size_t>(key_of(i)) + 1];
    for (std::size_t k = 1; k <= n_keys; ++k)
        csr.start[k] += csr.start[k - 1];
    csr.items.resize(n_items);
    std::vector<Index> next(csr.start.begin(), csr.start.end() - 1);
    for (std::size_t i = 0; i < n_items; ++i)
        csr.items[static_cast<std::size_t>(next[static_cast<std::size_t>(key_of(i))]++)] =
            value_of(i);
    return csr;
}

/// Elements incident to each node.
inline Csr node_elements(const Mesh& m) {
    const auto nv = static_cast<std::size_t>(m.vertices_per_element());
    return make_csr(
        m.pos.size(),
        m.con.size(),
        [&m](std::size_t i) { return m.con[i]; },
        [nv](std::size_t i) { return static_cast<Index>(i / nv); });
}

inline bool contains(std::span<const Index> element, Index v) {
    return std::find(element.begin(), element.end(), v) != element.end();
}

/// True if some element of `m` has both a and b as vertices, i.e. the edge.
inline bool has_edge(const Mesh& m, const Csr& adj, Index a, Index b) {
    for (Index e : adj.of(a))
        if (contains(m.element(e), b))
            return true;
    return false;
}

inline void validate(const Mesh& m) {
    if (m.dim != 2 && m.dim != 3)
        throw std::invalid_argument("simplex: dim must be 2 or 3, got " + std::to_string(m.dim));
    if (m.con.size() % static_cast<std::size_t>(m.vertices_per_element()) != 0)
        throw std::invalid_argument("simplex: con size is not a multiple of dim + 1");
    const auto n = static_cast<Index>(m.pos.size());
    const auto in_range = [n](Index v) { return v >= 0 && v < n; };
    if (!std::all_of(m.con.begin(), m.con.end(), in_range))
        throw std::out_of_range("simplex: element vertex out of range");
    for (const auto& h : m.hn)
        if (!std::all_of(h.begin(), h.end(), in_range))
            throw std::out_of_range("simplex: hanging-node row out of range");
}

inline Point midpoint(const Point& a, const Point& b) {
    return {0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.5 * (a[2] + b[2])};
}

/// Mean-ratio quality 12 (3|V|)^(2/3) / sum(l^2): 1 for a regular tetrahedron.
inline double tet_quality(const Point& a, const Point& b, const Point& c, const Point& d) {
    const auto sub = [](const Point& p, const Point& q) {
        return Point{p[0] - q[0], p[1] - q[1], p[2] - q[2]};
    };
    const auto dot = [](const Point& p, const Point& q) {
        return p[0] * q[0] + p[1] * q[1] + p[2] * q[2];
    };
    const Point u = sub(b, a), v = sub(c, a), w = sub(d, a);
    const Point uxv{
        u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
    const double volume = dot(uxv, w) / 6.0;
    const double sum_l2 = dot(u, u) + dot(v, v) + dot(w, w) + dot(sub(c, b), sub(c, b)) +
                          dot(sub(d, b), sub(d, b)) + dot(sub(d, c), sub(d, c));
    return 12.0 * std::pow(3.0 * std::abs(volume), 2.0 / 3.0) /
           std::max(sum_l2, std::numeric_limits<double>::epsilon());
}

// Local numbering of a refined element: corners 0..dim, then edge midpoints.
// Triangle: 3 = m01, 4 = m12, 5 = m02.
inline constexpr std::array<std::array<int, 2>, 3> kTriEdges{{{0, 1}, {1, 2}, {0, 2}}};
inline constexpr std::array<std::array<int, 3>, 4> kTriChildren{
    {{0, 3, 5}, {3, 1, 4}, {5, 4, 2}, {3, 4, 5}}};
// Tetrahedron: 4 = m01, 5 = m02, 6 = m03, 7 = m12, 8 = m13, 9 = m23.
inline constexpr std::array<std::array<int, 2>, 6> kTetEdges{
    {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}};
inline constexpr std::array<std::array<int, 4>, 4> kTetCorners{
    {{0, 4, 5, 6}, {4, 1, 7, 8}, {5, 7, 2, 9}, {6, 8, 9, 3}}};
// The octahedron's 4 children for each diagonal: m13-m02, m01-m23, m03-m12.
inline constexpr std::array<std::array<std::array<int, 4>, 4>, 3> kOctahedron{{
    {{{8, 4, 6, 5}, {8, 7, 4, 5}, {8, 9, 7, 5}, {8, 6, 9, 5}}},
    {{{4, 9, 6, 8}, {4, 9, 5, 6}, {4, 9, 7, 5}, {4, 9, 8, 7}}},
    {{{6, 7, 8, 4}, {6, 7, 4, 5}, {6, 7, 5, 9}, {6, 7, 9, 8}}},
}};

/// The octahedron split whose worst child quality is highest; first wins ties.
inline const std::array<std::array<int, 4>, 4>& best_octahedron_split(
    const std::array<Point, 10>& p) {
    std::size_t best = 0;
    double best_worst = -1.0;
    for (std::size_t k = 0; k < kOctahedron.size(); ++k) {
        double worst = std::numeric_limits<double>::infinity();
        for (const auto& t : kOctahedron[k])
            worst = std::min(worst,
                             tet_quality(p[static_cast<std::size_t>(t[0])],
                                         p[static_cast<std::size_t>(t[1])],
                                         p[static_cast<std::size_t>(t[2])],
                                         p[static_cast<std::size_t>(t[3])]));
        if (worst > best_worst) {
            best_worst = worst;
            best = k;
        }
    }
    return kOctahedron[best];
}

}  // namespace detail

/**
 * @brief Closes a refinement list under 1-irregular balance.
 *
 * Refining an element that has a hanging node H as a vertex while an element
 * across H's parent edge stays coarse would hang a new node on H: a chained,
 * 2-irregular node that master-slave condensation rejects, and that refine()
 * cannot track. So every element with H's parent edge but not H is added,
 * transitively. The result is the least closed superset of `elements`, the
 * list MATLAB's mesh_balance_refine_list returns.
 *
 * @return The closed list, ascending and without duplicates.
 */
[[nodiscard]] inline std::vector<Index> balance_closure(const Mesh& mesh,
                                                        std::span<const Index> elements) {
    detail::validate(mesh);
    const Index n_elem = mesh.num_elements();
    std::vector<char> selected(static_cast<std::size_t>(n_elem), 0);
    std::vector<Index> work;
    for (Index e : elements) {
        if (e < 0 || e >= n_elem)
            throw std::out_of_range("simplex: element index out of range");
        if (!selected[static_cast<std::size_t>(e)]) {
            selected[static_cast<std::size_t>(e)] = 1;
            work.push_back(e);
        }
    }

    if (!mesh.hn.empty()) {
        const auto adj = detail::node_elements(mesh);
        const auto rows_of = detail::make_csr(
            mesh.pos.size(),
            mesh.hn.size(),
            [&mesh](std::size_t i) { return mesh.hn[i][0]; },
            [](std::size_t i) { return static_cast<Index>(i); });
        while (!work.empty()) {
            const Index e = work.back();
            work.pop_back();
            for (Index h : mesh.element(e))
                for (Index r : rows_of.of(h)) {
                    const auto& row = mesh.hn[static_cast<std::size_t>(r)];
                    for (Index f : adj.of(row[1])) {
                        const auto vf = mesh.element(f);
                        if (!selected[static_cast<std::size_t>(f)] &&
                            detail::contains(vf, row[2]) && !detail::contains(vf, h)) {
                            selected[static_cast<std::size_t>(f)] = 1;
                            work.push_back(f);
                        }
                    }
                }
        }
    }

    std::vector<Index> closed;
    for (Index e = 0; e < n_elem; ++e)
        if (selected[static_cast<std::size_t>(e)])
            closed.push_back(e);
    return closed;
}

/**
 * @brief Red-refines the listed elements of `mesh`.
 *
 * @param mesh      T3/T4 mesh; `mesh.hn` lists its hanging nodes.
 * @param elements  Elements to refine (any order; duplicates are ignored),
 *   closed under balance_closure(). An unbalanced list can hang a node on
 *   a face; that node is not tracked and a later refinement duplicates it.
 * @return The refined mesh and its prolongation. Old nodes keep their
 *   indices; new nodes follow in order of first use. Each element is
 *   replaced in place by its children, in the child order above.
 */
[[nodiscard]] inline Refinement refine(const Mesh& mesh, std::span<const Index> elements) {
    detail::validate(mesh);
    const Index n_elem = mesh.num_elements();
    std::vector<char> marked(static_cast<std::size_t>(n_elem), 0);
    for (Index e : elements) {
        if (e < 0 || e >= n_elem)
            throw std::out_of_range("simplex: element index out of range");
        marked[static_cast<std::size_t>(e)] = 1;
    }
    const auto n_marked =
        static_cast<std::size_t>(std::count(marked.begin(), marked.end(), char{1}));
    const std::size_t n_children = mesh.dim == 2 ? 4 : 8;

    Refinement out;
    Mesh& m = out.mesh;
    m.dim = mesh.dim;
    m.pos = mesh.pos;
    m.con.reserve(mesh.con.size() + n_marked * (n_children - 1) *
                                        static_cast<std::size_t>(mesh.vertices_per_element()));

    std::unordered_map<std::uint64_t, Index> midpoint_of_edge;
    midpoint_of_edge.reserve(mesh.hn.size() + n_marked * (mesh.dim == 2 ? 3 : 6));
    for (const auto& h : mesh.hn)
        midpoint_of_edge.emplace(detail::edge_key(h[1], h[2]), h[0]);
    const auto midpoint_node = [&](Index a, Index b) {
        const auto [it, created] =
            midpoint_of_edge.try_emplace(detail::edge_key(a, b), static_cast<Index>(m.pos.size()));
        if (created) {
            if (m.pos.size() >= static_cast<std::size_t>(std::numeric_limits<Index>::max()))
                throw std::overflow_error("simplex: node count exceeds the index type");
            const Point p = detail::midpoint(m.pos[static_cast<std::size_t>(a)],
                                             m.pos[static_cast<std::size_t>(b)]);
            m.pos.push_back(p);
            out.parents.push_back({std::min(a, b), std::max(a, b)});
        }
        return it->second;
    };
    const auto emit = [&m](const auto& local, const auto& child) {
        for (int i : child)
            m.con.push_back(local[static_cast<std::size_t>(i)]);
    };

    for (Index e = 0; e < n_elem; ++e) {
        const auto v = mesh.element(e);
        if (!marked[static_cast<std::size_t>(e)]) {
            m.con.insert(m.con.end(), v.begin(), v.end());
            continue;
        }
        if (mesh.dim == 2) {
            std::array<Index, 6> local{v[0], v[1], v[2]};
            for (std::size_t i = 0; i < detail::kTriEdges.size(); ++i)
                local[3 + i] = midpoint_node(v[static_cast<std::size_t>(detail::kTriEdges[i][0])],
                                             v[static_cast<std::size_t>(detail::kTriEdges[i][1])]);
            for (const auto& child : detail::kTriChildren)
                emit(local, child);
        } else {
            std::array<Index, 10> local{v[0], v[1], v[2], v[3]};
            for (std::size_t i = 0; i < detail::kTetEdges.size(); ++i)
                local[4 + i] = midpoint_node(v[static_cast<std::size_t>(detail::kTetEdges[i][0])],
                                             v[static_cast<std::size_t>(detail::kTetEdges[i][1])]);
            std::array<Point, 10> p;
            for (std::size_t i = 0; i < p.size(); ++i)
                p[i] = m.pos[static_cast<std::size_t>(local[i])];
            for (const auto& child : detail::kTetCorners)
                emit(local, child);
            for (const auto& child : detail::best_octahedron_split(p))
                emit(local, child);
        }
    }

    // A midpoint hangs while some element still has its parent edge.
    const auto n_old = static_cast<Index>(mesh.pos.size());
    std::vector<std::array<Index, 3>> rows;
    rows.reserve(out.parents.size() + mesh.hn.size());
    for (std::size_t k = 0; k < out.parents.size(); ++k)
        rows.push_back({n_old + static_cast<Index>(k), out.parents[k][0], out.parents[k][1]});
    for (const auto& h : mesh.hn)
        rows.push_back({h[0], std::min(h[1], h[2]), std::max(h[1], h[2])});

    const auto adj = detail::node_elements(m);
    for (const auto& r : rows)
        if (detail::has_edge(m, adj, r[1], r[2]))
            m.hn.push_back(r);
    return out;
}

}  // namespace amr::simplex
