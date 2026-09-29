/**
 * @file test_util.hpp
 * @brief Mesh builders and independent oracles shared by the simplex test files.
 */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <numeric>
#include <random>
#include <vector>

#include "simplex.hpp"

namespace amr::simplex::test {

inline Point sub(const Point& a, const Point& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

/// Signed area (T3) or volume (T4) of element e.
inline double signed_measure(const Mesh& m, Index e) {
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

inline double total_measure(const Mesh& m) {
    double sum = 0;
    for (Index e = 0; e < m.num_elements(); ++e)
        sum += signed_measure(m, e);
    return sum;
}

/// Positively oriented element from its vertices.
inline void add_element(Mesh& m, std::vector<Index> v) {
    const auto first = m.con.size();
    m.con.insert(m.con.end(), v.begin(), v.end());
    if (signed_measure(m, m.num_elements() - 1) < 0)
        std::swap(m.con[first], m.con[first + 1]);
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

inline std::vector<Index> random_subset(Index n, double fraction, std::mt19937& rng) {
    std::vector<Index> all(static_cast<std::size_t>(n));
    std::iota(all.begin(), all.end(), 0);
    std::shuffle(all.begin(), all.end(), rng);
    all.resize(static_cast<std::size_t>(std::ceil(fraction * n)));
    return all;
}

/// The MATLAB loop, verbatim: sweep the hanging nodes until nothing changes.
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
inline Mesh randomly_refined(Mesh m, int cycles, double fraction, std::mt19937& rng) {
    for (int c = 0; c < cycles; ++c)
        m = refine(m, balance_closure(m, random_subset(m.num_elements(), fraction, rng))).mesh;
    return m;
}

/// Worst child quality of the best octahedron split, enumerating the three
/// diagonals (midpoints of opposite edges) and the equator around each.
inline double best_octahedron_worst_quality(const std::array<Point, 4>& c) {
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

}  // namespace amr::simplex::test
