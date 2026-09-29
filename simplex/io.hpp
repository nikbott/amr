/**
 * @file io.hpp
 * @brief SMX1: a simplicial mesh plus one index array, as one binary file.
 *
 * @details The exchange format between simplex.hpp and MATLAB until the MEX
 * interface exists; the reader and writer on the MATLAB side live in the
 * adaptive-dic tests that drive `amr_simplex` (cli.cpp).
 *
 * Layout, little-endian, indices 0-based:
 *   [ magic      : 4 bytes = "SMX1" ]
 *   [ dim        : uint32  (2 or 3) ]
 *   [ n_nodes, n_elements, n_hanging, n_indices : uint64 each ]
 *   [ pos        : float64 * 3 * n_nodes,         row-major ]
 *   [ con        : int32 * (dim + 1) * n_elements, row-major ]
 *   [ hn         : int32 * 3 * n_hanging,          row-major ]
 *   [ indices    : int32 * n_indices ]
 */
#pragma once

#include <array>
#include <bit>
#include <cstdint>
#include <fstream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "simplex.hpp"

namespace amr::simplex::io {

static_assert(std::endian::native == std::endian::little, "SMX1 is written in host byte order");
static_assert(sizeof(Point) == 3 * sizeof(double), "Point must be three packed doubles");

inline constexpr std::array<char, 4> kMagic{'S', 'M', 'X', '1'};

struct Payload {
    Mesh mesh;
    std::vector<Index> indices;
};

namespace detail {

template <class T>
void put(std::ofstream& out, std::span<const T> values) {
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size_bytes()));
}

template <class T>
void get(std::ifstream& in, std::span<T> values, const std::string& path) {
    in.read(reinterpret_cast<char*>(values.data()),
            static_cast<std::streamsize>(values.size_bytes()));
    if (!in)
        throw std::runtime_error("SMX1: " + path + " is truncated");
}

}  // namespace detail

inline void write(const std::string& path, const Mesh& mesh, std::span<const Index> indices) {
    std::ofstream out(path, std::ios::binary);
    if (!out)
        throw std::runtime_error("SMX1: cannot open " + path + " for writing");
    const auto dim = static_cast<std::uint32_t>(mesh.dim);
    const std::array<std::uint64_t, 4> counts{mesh.pos.size(),
                                              static_cast<std::uint64_t>(mesh.num_elements()),
                                              mesh.hn.size(),
                                              indices.size()};
    out.write(kMagic.data(), kMagic.size());
    detail::put(out, std::span<const std::uint32_t>(&dim, 1));
    detail::put(out, std::span<const std::uint64_t>(counts));
    detail::put(out, std::span<const Point>(mesh.pos));
    detail::put(out, std::span<const Index>(mesh.con));
    detail::put(out, std::span<const std::array<Index, 3>>(mesh.hn));
    detail::put(out, indices);
    if (!out)
        throw std::runtime_error("SMX1: failed writing " + path);
}

inline Payload read(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in)
        throw std::runtime_error("SMX1: cannot open " + path);
    const auto size = static_cast<std::uint64_t>(in.tellg());
    in.seekg(0);

    std::array<char, 4> magic{};
    std::uint32_t dim = 0;
    std::array<std::uint64_t, 4> counts{};
    detail::get(in, std::span<char>(magic), path);
    if (magic != kMagic)
        throw std::runtime_error("SMX1: " + path + " is not an SMX1 file");
    detail::get(in, std::span<std::uint32_t>(&dim, 1), path);
    if (dim != 2 && dim != 3)
        throw std::runtime_error("SMX1: dim must be 2 or 3");
    detail::get(in, std::span<std::uint64_t>(counts), path);

    // Check the sizes against the file before allocating anything.
    const auto [n_nodes, n_elements, n_hanging, n_indices] = counts;
    constexpr std::uint64_t kMax = static_cast<std::uint64_t>(std::numeric_limits<Index>::max());
    if (n_nodes > kMax || n_elements > kMax || n_hanging > kMax || n_indices > kMax)
        throw std::runtime_error("SMX1: counts exceed the index type");
    const std::uint64_t expected =
        kMagic.size() + sizeof dim + sizeof counts + n_nodes * sizeof(Point) +
        (n_elements * (dim + 1) + 3 * n_hanging + n_indices) * sizeof(Index);
    if (size != expected)
        throw std::runtime_error("SMX1: " + path + " has " + std::to_string(size) +
                                 " bytes, expected " + std::to_string(expected));

    Payload p;
    p.mesh.dim = static_cast<int>(dim);
    p.mesh.pos.resize(n_nodes);
    p.mesh.con.resize(n_elements * (dim + 1));
    p.mesh.hn.resize(n_hanging);
    p.indices.resize(n_indices);
    detail::get(in, std::span<Point>(p.mesh.pos), path);
    detail::get(in, std::span<Index>(p.mesh.con), path);
    detail::get(in, std::span<std::array<Index, 3>>(p.mesh.hn), path);
    detail::get(in, std::span<Index>(p.indices), path);
    return p;
}

}  // namespace amr::simplex::io
