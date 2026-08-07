/**
 * @file common/viz.hpp
 * @brief Shared mesh writers: SVG (2D) and VTK legacy UNSTRUCTURED_GRID (2D/3D).
 *
 * Single source of truth for visualization, in the same spirit as
 * common/core.hpp: the omp/, mpi/ and cuda/ backends each used to carry their
 * own copy, and the copies had diverged (one emitted a VTK_VERTEX point cloud
 * instead of cells; another declared VTK_VOXEL while writing hexahedron corner
 * ordering). The writers here take a backend-agnostic LeafView of
 * `(codes, levels)`; each backend keeps only a thin adapter that fills it in.
 *
 * Both formats carry the refinement level per cell, so the grading of the mesh
 * is directly inspectable: colour by `level` in ParaView, or read the fill
 * colour in the SVG.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "core.hpp"  // amr::MortonCode, Coordinate, morton::

namespace amr::viz {

/// A backend-agnostic view of a leaf set. Non-owning; valid for the call only.
struct LeafView {
    const uint64_t* codes = nullptr;
    const uint8_t* levels = nullptr;
    std::size_t n = 0;
    int max_level = 0;
    int dim = 2;
};

namespace detail {

/// Decode leaf `i` into integer corner coordinates (unused axes read back 0).
inline void decode_leaf(const LeafView& v, std::size_t i, Coordinate c[3]) {
    c[0] = c[1] = c[2] = Coordinate{0};
    if (v.dim == 2)
        morton::decode_2d(MortonCode{v.codes[i]}, c[0], c[1]);
    else
        morton::decode_3d(MortonCode{v.codes[i]}, c[0], c[1], c[2]);
}

inline void check(const LeafView& v, const char* who) {
    if (v.dim != 2 && v.dim != 3)
        throw std::runtime_error(std::string(who) + ": dim must be 2 or 3");
    if (v.n && (!v.codes || !v.levels))
        throw std::runtime_error(std::string(who) + ": null leaf arrays");
}

inline std::ofstream open_or_throw(const std::string& path, const char* who) {
    std::ofstream f(path);
    if (!f)
        throw std::runtime_error(std::string(who) + ": cannot open " + path);
    return f;
}

/// Level -> hue on a blue(coarse) .. red(fine) ramp, so refinement reads at a glance.
inline int level_hue(int level, int lmin, int lmax) {
    if (lmax <= lmin)
        return 210;
    const double t = static_cast<double>(level - lmin) / static_cast<double>(lmax - lmin);
    return static_cast<int>(210.0 - 210.0 * t);  // 210 = blue, 0 = red
}

}  // namespace detail

/// Write a 2D leaf set as an SVG, each cell filled by its refinement level.
inline void write_svg(const std::string& path, const LeafView& v) {
    detail::check(v, "viz::write_svg");
    if (v.dim != 2)
        throw std::runtime_error("viz::write_svg: SVG output requires dim == 2");

    int lmin = v.max_level, lmax = 0;
    for (std::size_t i = 0; i < v.n; ++i) {
        const int l = static_cast<int>(v.levels[i]);
        if (l < lmin)
            lmin = l;
        if (l > lmax)
            lmax = l;
    }

    std::ofstream f = detail::open_or_throw(path, "viz::write_svg");
    const double scale = 1000.0 / static_cast<double>(1ULL << v.max_level);

    f << "<svg width=\"800\" height=\"800\" viewBox=\"0 0 1000 1000\" "
         "xmlns=\"http://www.w3.org/2000/svg\">\n";
    f << "<rect width=\"1000\" height=\"1000\" fill=\"white\"/>\n";

    for (std::size_t i = 0; i < v.n; ++i) {
        Coordinate c[3];
        detail::decode_leaf(v, i, c);
        const int level = static_cast<int>(v.levels[i]);
        const double s = static_cast<double>(1ULL << (v.max_level - level)) * scale;
        const double x = static_cast<double>(c[0].value) * scale;
        const double y = 1000.0 - static_cast<double>(c[1].value) * scale;  // flip Y

        f << "<rect x=\"" << x << "\" y=\"" << (y - s) << "\" width=\"" << s << "\" height=\"" << s
          << "\" fill=\"hsl(" << detail::level_hue(level, lmin, lmax)
          << ",75%,60%)\" stroke=\"black\" stroke-width=\"0.4\"/>\n";
    }
    f << "</svg>\n";
    std::cout << "Wrote " << path << " (" << v.n << " cells, levels " << lmin << ".." << lmax
              << ")\n";
}

/// Write a leaf set as VTK legacy UNSTRUCTURED_GRID: VTK_QUAD (2D) / VTK_HEXAHEDRON (3D).
inline void write_vtk(const std::string& path, const LeafView& v) {
    detail::check(v, "viz::write_vtk");

    const std::size_t corners = (v.dim == 2) ? 4u : 8u;
    const int cell_type = (v.dim == 2) ? 9 : 12;  // VTK_QUAD / VTK_HEXAHEDRON

    // Corner offsets in VTK winding order: base face counter-clockwise, then top.
    static constexpr int OFF[8][3] = {
        {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};

    std::ofstream f = detail::open_or_throw(path, "viz::write_vtk");
    const double norm = 1.0 / static_cast<double>(1ULL << v.max_level);

    f << "# vtk DataFile Version 3.0\nAMR Mesh\nASCII\nDATASET UNSTRUCTURED_GRID\n";

    f << "POINTS " << v.n * corners << " double\n";
    for (std::size_t i = 0; i < v.n; ++i) {
        Coordinate c[3];
        detail::decode_leaf(v, i, c);
        const uint64_t s = 1ULL << (v.max_level - static_cast<int>(v.levels[i]));
        for (std::size_t k = 0; k < corners; ++k) {
            for (int axis = 0; axis < 3; ++axis) {
                const double p =
                    (axis < v.dim) ? static_cast<double>(c[axis].value +
                                                         static_cast<uint64_t>(OFF[k][axis]) * s) *
                                         norm
                                   : 0.0;
                f << p << (axis == 2 ? '\n' : ' ');
            }
        }
    }

    f << "\nCELLS " << v.n << " " << v.n * (corners + 1) << "\n";
    for (std::size_t i = 0; i < v.n; ++i) {
        f << corners;
        for (std::size_t k = 0; k < corners; ++k)
            f << " " << i * corners + k;
        f << "\n";
    }

    f << "\nCELL_TYPES " << v.n << "\n";
    for (std::size_t i = 0; i < v.n; ++i)
        f << cell_type << "\n";

    f << "\nCELL_DATA " << v.n << "\nSCALARS level int 1\nLOOKUP_TABLE default\n";
    for (std::size_t i = 0; i < v.n; ++i)
        f << static_cast<int>(v.levels[i]) << "\n";

    std::cout << "Wrote " << path << " (" << v.n << " cells)\n";
}

}  // namespace amr::viz
