/**
 * @file structured.hpp
 * @brief Structured T3/T4 seed meshes on a box, as the FE-DIC solver seeds them.
 *
 * @details `structured(dims, nodes, origin)` spans [origin, origin + dims] with
 * nodes[k] equally spaced nodes on axis k and splits every grid cell into
 * simplices. It reproduces the MATLAB `mesh.genMesh` it replaces, up to
 * numbering:
 *
 * - Axis coordinates are MATLAB's `linspace(a, b, n)`: `a + (i * (b - a)) / (n
 *   - 1)`, with both ends set exactly.
 * - 3D: every cell is split into the same 6 tetrahedra, listed by corner
 *   (xyz, 1 = high side) in genMesh's vertex order: {000,110,010,011},
 *   {001,111,101,100}, {000,110,011,111}, {000,100,110,111},
 *   {000,001,100,111}, {000,011,001,111}. The split is
 *   translation invariant, so every face shared by two cells is cut along the
 *   same diagonal from both sides (xy and yz faces from the low corner to the
 *   high one, xz faces along the other diagonal).
 * - 2D: every cell is split along a diagonal chosen per quadrant: cells in the
 *   low half of both axes, or the high half of both (half = the first
 *   floor(cells / 2)), take the diagonal from the low corner to the high one;
 *   the others take the other diagonal (so the diagonals point towards the
 *   centre when the cell counts are even).
 *
 * Nodes and cells are numbered as genMesh numbers them: y fastest, then x,
 * then z. All elements are positively oriented.
 */
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include "simplex.hpp"

namespace amr::simplex {

namespace detail {

/// MATLAB's linspace(a, b, n) for n >= 2, bit for bit, including its
/// branches for spans whose intermediate products overflow.
inline std::vector<double> linspace(double a, double b, Index n) {
    std::vector<double> y(static_cast<std::size_t>(n));
    const double n1 = static_cast<double>(n - 1);
    const bool overflows = std::isinf((b - a) * (n1 - 1.0));
    for (Index i = 0; i < n; ++i) {
        const double t = static_cast<double>(i);
        double v;
        if (!overflows)
            v = a + (t * (b - a)) / n1;
        else if (std::isinf(b - a))  // opposite signs overflow
            v = a + (b / n1) * t - (a / n1) * t;
        else
            v = a + t * ((b - a) / n1);
        y[static_cast<std::size_t>(i)] = v;
    }
    y.front() = a;
    y.back() = b;
    if (a == b)
        std::fill(y.begin(), y.end(), a);
    return y;
}

// The 6 tetrahedra of a cell as corner indices x | y << 1 | z << 2, in the
// vertex order genMesh emits them.
inline constexpr std::array<std::array<int, 4>, 6> kCellTets{{
    {0b000, 0b011, 0b010, 0b110},
    {0b100, 0b111, 0b101, 0b001},
    {0b000, 0b011, 0b110, 0b111},
    {0b000, 0b001, 0b011, 0b111},
    {0b000, 0b100, 0b001, 0b111},
    {0b000, 0b110, 0b100, 0b111},
}};

}  // namespace detail

[[nodiscard]] inline Mesh structured(std::span<const double> dims,
                                     std::span<const Index> nodes,
                                     std::span<const double> origin) {
    const std::size_t d = dims.size();
    if ((d != 2 && d != 3) || nodes.size() != d || origin.size() != d)
        throw std::invalid_argument(
            "structured: dims, nodes and origin must all have 2 or 3 entries");
    std::int64_t points = 1, cells = 1;
    for (std::size_t k = 0; k < d; ++k) {
        if (nodes[k] < 2)
            throw std::invalid_argument("structured: every axis needs at least 2 nodes");
        if (!(std::isfinite(dims[k]) && dims[k] > 0.0 && std::isfinite(origin[k])))
            throw std::invalid_argument(
                "structured: dims must be finite and positive, origin finite");
        points *= nodes[k];  // each factor fits in 31 bits, so checking every step cannot overflow
        cells *= nodes[k] - 1;
        if (points > std::numeric_limits<Index>::max() ||
            cells * (d == 2 ? 2 : 6) > std::numeric_limits<Index>::max())
            throw std::overflow_error("structured: mesh exceeds the index type");
    }

    std::array<std::vector<double>, 3> axis;
    for (std::size_t k = 0; k < d; ++k)
        axis[k] = detail::linspace(origin[k], origin[k] + dims[k], nodes[k]);
    if (d == 2)
        axis[2] = {0.0};
    const std::array<Index, 3> n{nodes[0], nodes[1], d == 3 ? nodes[2] : 1};
    const auto id = [&n](Index i, Index j, Index k) { return (k * n[0] + i) * n[1] + j; };

    Mesh m;
    m.dim = static_cast<int>(d);
    for (Index k = 0; k < n[2]; ++k)
        for (Index i = 0; i < n[0]; ++i)
            for (Index j = 0; j < n[1]; ++j)
                m.pos.push_back({axis[0][static_cast<std::size_t>(i)],
                                 axis[1][static_cast<std::size_t>(j)],
                                 axis[2][static_cast<std::size_t>(k)]});

    if (d == 2) {
        const Index half_x = (n[0] - 1) / 2, half_y = (n[1] - 1) / 2;
        for (Index i = 0; i + 1 < n[0]; ++i)
            for (Index j = 0; j + 1 < n[1]; ++j) {
                const Index a = id(i, j, 0), b = id(i + 1, j, 0);          // bottom edge
                const Index c = id(i + 1, j + 1, 0), e = id(i, j + 1, 0);  // top edge
                if ((i < half_x) == (j < half_y))
                    m.con.insert(m.con.end(), {a, b, c, c, e, a});  // diagonal a-c
                else
                    m.con.insert(m.con.end(), {a, b, e, b, c, e});  // diagonal b-e
            }
        return m;
    }
    for (Index k = 0; k + 1 < n[2]; ++k)
        for (Index i = 0; i + 1 < n[0]; ++i)
            for (Index j = 0; j + 1 < n[1]; ++j)
                for (const auto& tet : detail::kCellTets)
                    for (int bits : tet)
                        m.con.push_back(
                            id(i + (bits & 1), j + ((bits >> 1) & 1), k + ((bits >> 2) & 1)));
    return m;
}

}  // namespace amr::simplex
