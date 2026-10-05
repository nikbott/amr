/**
 * @file simplex.hpp
 * @brief Red refinement of unstructured and hybrid meshes with hanging nodes
 * and 1-irregular balance.
 *
 * @details The unstructured counterpart of the Morton octree. Its meshes may
 * mix any shapes of element.hpp of one dimension: triangles and
 * quadrilaterals in 2D; tetrahedra, hexahedra, prisms and pyramids in 3D. The
 * refinement itself is element-agnostic; each shape's reference element says
 * how it splits:
 *
 * - A refined element's new nodes are its template's lattice nodes, each a
 *   weighted mean of the element's vertices. A node is keyed by its parents,
 *   so neighbours sharing an edge or a face share its nodes, and a listed
 *   hanging node is reused rather than duplicated.
 * - A shape with several splits (the tetrahedron's 3 octahedron diagonals
 *   [Bey1995]) takes the one whose worst child has the highest mean-ratio
 *   quality [Zhang1995]; ties go to the earliest split.
 * - A node hangs while some element still has all of its parents, i.e. the
 *   unrefined edge or face it sits on. Its constraint is its prolongation row.
 *   Lists are closed under balance_closure() first, which keeps the mesh
 *   1-irregular: no hanging node has a hanging parent.
 *
 * On triangle and tetrahedron meshes this reproduces the FE-DIC solver's
 * MATLAB `mesh_subdivide_list` and `mesh_balance_refine_list` exactly, node
 * and element numbering included, since correlation round-off depends on
 * them. A new node's coordinates are the weighted sum of its parents' in
 * ascending parent order: for a midpoint exactly 0.5 * (a + b), the value
 * MATLAB's S * pos produces.
 */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "element.hpp"

namespace amr::simplex {

using Index = std::int32_t;

/// Node `node` as a weighted sum of its parents: a hanging node's constraint,
/// or a new node's prolongation row.
struct Constraint {
    Index node = 0;
    std::vector<Index> parent;   ///< ascending in the engine's output
    std::vector<double> weight;  ///< one per parent, summing to 1
    bool operator==(const Constraint&) const = default;
};

/// Unstructured mesh of shapes of one dimension. Indices are 0-based.
struct Mesh {
    int dim = 2;
    std::vector<Point> pos;              ///< z = 0 in 2D, as in the DIC meshes
    std::vector<Shape> type;             ///< one per element
    std::vector<std::size_t> offset{0};  ///< element e's vertices: con[offset[e] .. offset[e + 1])
    std::vector<Index> con;              ///< vertices in each shape's node order
    std::vector<Constraint> hn;          ///< hanging nodes

    [[nodiscard]] Index num_elements() const { return static_cast<Index>(type.size()); }
    [[nodiscard]] std::span<const Index> element(Index e) const {
        const auto k = static_cast<std::size_t>(e);
        return {con.data() + offset[k], offset[k + 1] - offset[k]};
    }
    void add(Shape s, std::span<const Index> vertices) {
        if (type.size() >= static_cast<std::size_t>(std::numeric_limits<Index>::max()))
            throw std::overflow_error("simplex: element count exceeds the index type");
        type.push_back(s);
        con.insert(con.end(), vertices.begin(), vertices.end());
        offset.push_back(con.size());
    }
    void add(Shape s, std::initializer_list<Index> vertices) {
        add(s, std::span<const Index>(vertices.begin(), vertices.size()));
    }
};

/// A refined mesh and its prolongation U_new = S U_old: node i < n_old keeps
/// its value, and new node n_old + k is prolongation[k].
struct Refinement {
    Mesh mesh;
    std::vector<Constraint> prolongation;
};

namespace detail {

/// Items grouped by an integer key, in compressed-row form. Offsets are
/// size_t: connectivity, and so the items, can exceed 2^31 entries.
struct Csr {
    std::vector<std::size_t> start;  ///< size n_keys + 1
    std::vector<Index> items;
    [[nodiscard]] std::span<const Index> of(Index key) const {
        const auto b = start[static_cast<std::size_t>(key)];
        const auto e = start[static_cast<std::size_t>(key) + 1];
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
    std::vector<std::size_t> next(csr.start.begin(), csr.start.end() - 1);
    for (std::size_t i = 0; i < n_items; ++i)
        csr.items[next[static_cast<std::size_t>(key_of(i))]++] = value_of(i);
    return csr;
}

/// Elements incident to each node.
inline Csr node_elements(const Mesh& m) {
    std::vector<Index> element_of(m.con.size());
    for (Index e = 0; e < m.num_elements(); ++e)
        for (auto i = m.offset[static_cast<std::size_t>(e)];
             i < m.offset[static_cast<std::size_t>(e) + 1];
             ++i)
            element_of[i] = e;
    return make_csr(
        m.pos.size(),
        m.con.size(),
        [&m](std::size_t i) { return m.con[i]; },
        [&element_of](std::size_t i) { return element_of[i]; });
}

inline bool contains(std::span<const Index> element, Index v) {
    return std::find(element.begin(), element.end(), v) != element.end();
}

/// True if some element of `m` has every one of `nodes` as a vertex.
inline bool has_all(const Mesh& m, const Csr& adj, std::span<const Index> nodes) {
    for (Index e : adj.of(nodes[0])) {
        const auto v = m.element(e);
        if (std::all_of(nodes.begin() + 1, nodes.end(), [&v](Index n) { return contains(v, n); }))
            return true;
    }
    return false;
}

/// Parents of a node, sorted, padded with -1: the key that makes neighbours
/// share it. No element has more than 8 vertices, so no node more parents.
using NodeKey = std::array<Index, 8>;

inline void validate(const Mesh& m) {
    if (m.dim != 2 && m.dim != 3)
        throw std::invalid_argument("simplex: dim must be 2 or 3, got " + std::to_string(m.dim));
    constexpr auto kMax = static_cast<std::size_t>(std::numeric_limits<Index>::max());
    if (m.type.size() > kMax || m.pos.size() > kMax)
        throw std::overflow_error("simplex: mesh exceeds the index type");
    if (m.offset.size() != m.type.size() + 1 || m.offset.front() != 0 ||
        m.offset.back() != m.con.size())
        throw std::invalid_argument("simplex: offset does not match type and con");
    for (std::size_t e = 0; e < m.type.size(); ++e) {
        const auto& t = element_type(m.type[e]);
        if (t.dim != m.dim)
            throw std::invalid_argument("simplex: element of another dimension");
        if (m.offset[e + 1] - m.offset[e] != static_cast<std::size_t>(t.vertices))
            throw std::invalid_argument("simplex: offset does not match the element's shape");
    }
    const auto n = static_cast<Index>(m.pos.size());
    const auto in_range = [n](Index v) { return v >= 0 && v < n; };
    if (!std::all_of(m.con.begin(), m.con.end(), in_range))
        throw std::out_of_range("simplex: element vertex out of range");
    NodeKey sorted;
    std::vector<char> constrained(m.pos.size(), 0);
    for (const auto& h : m.hn) {
        if (h.parent.size() < 2 || h.parent.size() > sorted.size() ||
            h.parent.size() != h.weight.size())
            throw std::invalid_argument(
                "simplex: a constraint needs 2 to 8 parents, with one weight each");
        if (!in_range(h.node) || !std::all_of(h.parent.begin(), h.parent.end(), in_range))
            throw std::out_of_range("simplex: constraint node out of range");
        const auto end = std::copy(h.parent.begin(), h.parent.end(), sorted.begin());
        std::sort(sorted.begin(), end);
        if (std::adjacent_find(sorted.begin(), end) != end ||
            std::binary_search(sorted.begin(), end, h.node))
            throw std::invalid_argument("simplex: a constraint's parents must be distinct nodes");
        double sum = 0.0;
        for (double w : h.weight)
            sum += w;
        if (!(std::abs(sum - 1.0) <= 1e-12))  // also rejects non-finite weights
            throw std::invalid_argument("simplex: constraint weights must sum to 1");
        if (constrained[static_cast<std::size_t>(h.node)]++)
            throw std::invalid_argument("simplex: a node is constrained twice");
        // The node must sit where its parents' refinement puts it: refine()
        // reuses it for that lattice point and builds children on it.
        const auto x = [&m](Index v) { return m.pos[static_cast<std::size_t>(v)]; };
        for (std::size_t c = 0; c < 3; ++c) {
            double mean = 0.0, scale = 1.0;
            for (std::size_t k = 0; k < h.parent.size(); ++k) {
                mean += h.weight[k] * x(h.parent[k])[c];
                scale = std::max(scale, std::abs(x(h.parent[k])[c]));
            }
            if (!(std::abs(x(h.node)[c] - mean) <= 1e-9 * scale))
                throw std::invalid_argument(
                    "simplex: a hanging node is not at its parents' weighted mean");
        }
    }
}

struct NodeKeyHash {
    std::size_t operator()(const NodeKey& k) const {
        std::uint64_t h = 1469598103934665603ULL;  // FNV-1a
        for (Index v : k) {
            h ^= static_cast<std::uint32_t>(v);
            h *= 1099511628211ULL;
        }
        return static_cast<std::size_t>(h);
    }
};

/// Sorts a row's parents ascending, carrying the weights along.
inline void sort_row(Constraint& row) {
    const std::size_t n = row.parent.size();
    for (std::size_t i = 1; i < n; ++i)  // insertion sort: rows hold a few parents
        for (std::size_t j = i; j > 0 && row.parent[j] < row.parent[j - 1]; --j) {
            std::swap(row.parent[j], row.parent[j - 1]);
            std::swap(row.weight[j], row.weight[j - 1]);
        }
}

/// The key of a row with sorted parents, at most 8 of them (validate()).
inline NodeKey key_of(const Constraint& sorted) {
    NodeKey k;
    k.fill(-1);
    std::copy(sorted.parent.begin(), sorted.parent.end(), k.begin());
    return k;
}

}  // namespace detail

/**
 * @brief Closes a refinement list under 1-irregular balance.
 *
 * Refining an element that has a hanging node H as a vertex while an element
 * with all of H's parents (the unrefined edge or face H sits on) stays coarse
 * would hang a new node on H: a chained, 2-irregular node that master-slave
 * condensation rejects, and that refine() cannot track. So every element
 * with all of H's parents is added, transitively (none has H as well: it
 * would be degenerate). The result is the least closed superset of
 * `elements`, the list MATLAB's mesh_balance_refine_list returns.
 *
 * @return The closed list, ascending and without duplicates.
 */
namespace detail {
inline std::vector<Index> closure(const Mesh& mesh, std::span<const Index> elements);
inline Refinement refine_valid(const Mesh& mesh, std::span<const Index> elements);
}  // namespace detail

[[nodiscard]] inline std::vector<Index> balance_closure(const Mesh& mesh,
                                                        std::span<const Index> elements) {
    detail::validate(mesh);
    return detail::closure(mesh, elements);
}

/// The closure of a validated mesh (see balance_closure).
inline std::vector<Index> detail::closure(const Mesh& mesh, std::span<const Index> elements) {
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
            [&mesh](std::size_t i) { return mesh.hn[i].node; },
            [](std::size_t i) { return static_cast<Index>(i); });
        while (!work.empty()) {
            const Index e = work.back();
            work.pop_back();
            for (Index h : mesh.element(e))
                for (Index r : rows_of.of(h)) {
                    const auto& parent = mesh.hn[static_cast<std::size_t>(r)].parent;
                    for (Index f : adj.of(parent[0])) {
                        const auto vf = mesh.element(f);
                        if (!selected[static_cast<std::size_t>(f)] &&
                            std::all_of(parent.begin() + 1, parent.end(), [&vf](Index p) {
                                return detail::contains(vf, p);
                            })) {
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
 * @param mesh      Mesh of one dimension; `mesh.hn` lists its hanging nodes.
 * @param elements  Elements to refine (any order; duplicates are ignored),
 *   closed under balance_closure(). An unbalanced list can hang a node on a
 *   hanging node; that node is not tracked and a later refinement
 *   duplicates it.
 * @return The refined mesh and its prolongation. Old nodes keep their
 *   indices and new nodes follow in order of creation. The children of the
 *   refined elements come first, parent by parent in ascending order and in
 *   their template's order, then the other elements in their order. Hanging
 *   nodes list the new ones first, then the old ones still hanging, with
 *   parents ascending.
 */
[[nodiscard]] inline Refinement refine(const Mesh& mesh, std::span<const Index> elements) {
    detail::validate(mesh);
    return detail::refine_valid(mesh, elements);
}

/// The refinement of a validated mesh (see refine).
inline Refinement detail::refine_valid(const Mesh& mesh, std::span<const Index> elements) {
    const Index n_elem = mesh.num_elements();
    std::vector<char> marked(static_cast<std::size_t>(n_elem), 0);
    for (Index e : elements) {
        if (e < 0 || e >= n_elem)
            throw std::out_of_range("simplex: element index out of range");
        marked[static_cast<std::size_t>(e)] = 1;
    }

    Refinement out;
    Mesh& m = out.mesh;
    m.dim = mesh.dim;
    m.pos = mesh.pos;
    std::size_t n_children = 0, n_con = 0, n_new = 0;
    for (Index e = 0; e < n_elem; ++e)
        if (marked[static_cast<std::size_t>(e)]) {
            const auto& red = element_type(mesh.type[static_cast<std::size_t>(e)]).red;
            for (const auto* part : {&red.head, &red.choices.front(), &red.tail})
                for (const auto& c : *part) {
                    ++n_children;
                    n_con += c.node.size();
                }
            n_new += red.new_nodes.size();
        }
    m.type.reserve(mesh.type.size() + n_children);
    m.offset.reserve(mesh.offset.size() + n_children);
    m.con.reserve(mesh.con.size() + n_con);

    // Nodes by their parents: the listed hanging nodes (row >= 0 in `old`),
    // then the new ones (row -1, prolongation[node - n_old]).
    struct Known {
        Index node;
        Index row;
    };
    const auto n_old = static_cast<Index>(mesh.pos.size());
    std::vector<Constraint> old = mesh.hn;
    std::unordered_map<detail::NodeKey, Known, detail::NodeKeyHash> node_of;
    node_of.reserve(old.size() + n_new);
    for (std::size_t r = 0; r < old.size(); ++r) {
        detail::sort_row(old[r]);
        node_of.emplace(detail::key_of(old[r]), Known{old[r].node, static_cast<Index>(r)});
    }
    Constraint row;
    const auto new_node = [&](const LatticeNode& lattice, std::span<const Index> v) {
        row.parent.clear();
        row.weight = lattice.weight;
        for (int i : lattice.vertex)
            row.parent.push_back(v[static_cast<std::size_t>(i)]);
        detail::sort_row(row);
        const auto [it, created] =
            node_of.try_emplace(detail::key_of(row), Known{static_cast<Index>(m.pos.size()), -1});
        const Index node = it->second.node;
        if (!created) {  // a node that exists must be the lattice point asked for
            const auto& weight =
                it->second.row >= 0
                    ? old[static_cast<std::size_t>(it->second.row)].weight
                    : out.prolongation[static_cast<std::size_t>(node - n_old)].weight;
            if (weight != row.weight)
                throw std::invalid_argument("simplex: hanging node " + std::to_string(node) +
                                            " is not where its parents' refinement puts it");
            return node;
        }
        if (m.pos.size() >= static_cast<std::size_t>(std::numeric_limits<Index>::max()))
            throw std::overflow_error("simplex: node count exceeds the index type");
        const auto x = [&m](Index n) { return m.pos[static_cast<std::size_t>(n)]; };
        Point p = x(row.parent[0]);
        for (auto& c : p)
            c *= row.weight[0];
        for (std::size_t k = 1; k < row.parent.size(); ++k)
            for (std::size_t c = 0; c < 3; ++c)
                p[c] += row.weight[k] * x(row.parent[k])[c];
        m.pos.push_back(p);
        row.node = node;
        out.prolongation.push_back(row);
        return node;
    };

    std::vector<Index> local, child;
    std::vector<Point> at;
    for (Index e = 0; e < n_elem; ++e) {
        if (!marked[static_cast<std::size_t>(e)])
            continue;
        const auto v = mesh.element(e);
        const auto& red = element_type(mesh.type[static_cast<std::size_t>(e)]).red;
        local.assign(v.begin(), v.end());
        for (const auto& lattice : red.new_nodes)
            local.push_back(new_node(lattice, v));
        std::size_t choice = 0;
        if (red.choices.size() > 1) {
            at.clear();
            for (Index n : local)
                at.push_back(m.pos[static_cast<std::size_t>(n)]);
            choice = split_choice(red, at);
        }
        for (const auto* part : {&red.head, &red.choices[choice], &red.tail})
            for (const auto& c : *part) {
                child.clear();
                for (int n : c.node)
                    child.push_back(local[static_cast<std::size_t>(n)]);
                m.add(c.shape, child);
            }
    }
    for (Index e = 0; e < n_elem; ++e)
        if (!marked[static_cast<std::size_t>(e)])
            m.add(mesh.type[static_cast<std::size_t>(e)], mesh.element(e));

    // A node hangs while some element still has all of its parents.
    const auto adj = detail::node_elements(m);
    for (const auto& r : out.prolongation)
        if (detail::has_all(m, adj, r.parent))
            m.hn.push_back(r);
    for (auto& h : old)
        if (detail::has_all(m, adj, h.parent))
            m.hn.push_back(std::move(h));
    return out;
}

}  // namespace amr::simplex
