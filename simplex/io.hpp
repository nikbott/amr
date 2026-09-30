/**
 * @file io.hpp
 * @brief SMX2: a typed unstructured mesh plus named arrays, as one binary file.
 *
 * @details The exchange format between the engine and MATLAB; the MATLAB
 * reader and writer are adaptive-dic's `+amr` package. Little-endian, indices
 * 0-based:
 *
 *   [ magic  : 4 bytes = "SMX2" ]
 *   [ dim    : uint32 ]                      spatial dimension, 2 or 3
 *   [ n_nodes, n_elements, n_constraints, n_constraint_parents, n_fields : uint64 each ]
 *   [ pos    : float64 * 3 * n_nodes ]       row-major, z = 0 in 2D
 *   [ type   : uint8 * n_elements ]          Shape codes (element.hpp)
 *   [ con    : int32 * (vertices of each element, concatenated) ]
 *   [ constrained node : int32 * n_constraints ]
 *   [ parent count     : int32 * n_constraints ]
 *   [ parents          : int32 * n_constraint_parents ]
 *   [ weights          : float64 * n_constraint_parents ]
 *   per field:
 *   [ name length : uint32 ][ name ][ dtype : uint8 (0 = int32, 1 = float64) ]
 *   [ count : uint64 ][ values ]
 *
 * A constraint row gives a hanging node's value as a weighted sum of its
 * parents'. Element offsets follow from the types, so they are not stored.
 * read() checks the layout: known shapes of the file's dimension and
 * constraint counts that add up. The engine validates the mesh itself.
 */
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "simplex.hpp"

namespace amr::simplex::io {

static_assert(std::endian::native == std::endian::little, "SMX2 is written in host byte order");
static_assert(sizeof(Point) == 3 * sizeof(double), "Point must be three packed doubles");

inline constexpr std::array<char, 4> kMagic{'S', 'M', 'X', '2'};

struct Payload {
    Mesh mesh;
    std::map<std::string, std::vector<Index>> ints;
    std::map<std::string, std::vector<double>> doubles;
};

namespace detail {

template <class T>
void put(std::ofstream& out, std::span<const T> values) {
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size_bytes()));
}

/// Reads `count` values after checking they fit in what is left of the file.
template <class T>
std::vector<T> take(std::ifstream& in,
                    std::uint64_t count,
                    std::uint64_t& left,
                    const std::string& path) {
    if (count > left / sizeof(T))
        throw std::runtime_error("SMX2: " + path + " is truncated or its counts are corrupt");
    left -= count * sizeof(T);
    std::vector<T> v(static_cast<std::size_t>(count));
    in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(count * sizeof(T)));
    if (!in)
        throw std::runtime_error("SMX2: " + path + " is truncated");
    return v;
}

template <class T>
T take_one(std::ifstream& in, std::uint64_t& left, const std::string& path) {
    return take<T>(in, 1, left, path)[0];
}

}  // namespace detail

inline void write(const std::string& path,
                  const Mesh& mesh,
                  const std::map<std::string, std::vector<Index>>& ints = {},
                  const std::map<std::string, std::vector<double>>& doubles = {}) {
    std::ofstream out(path, std::ios::binary);
    if (!out)
        throw std::runtime_error("SMX2: cannot open " + path + " for writing");
    const auto dim = static_cast<std::uint32_t>(mesh.dim);
    std::vector<Index> node, count, parent;
    std::vector<double> weight;
    for (const auto& h : mesh.hn) {
        node.push_back(h.node);
        count.push_back(static_cast<Index>(h.parent.size()));
        parent.insert(parent.end(), h.parent.begin(), h.parent.end());
        weight.insert(weight.end(), h.weight.begin(), h.weight.end());
    }
    const std::array<std::uint64_t, 5> counts{mesh.pos.size(),
                                              mesh.type.size(),
                                              mesh.hn.size(),
                                              parent.size(),
                                              ints.size() + doubles.size()};
    out.write(kMagic.data(), kMagic.size());
    detail::put(out, std::span<const std::uint32_t>(&dim, 1));
    detail::put(out, std::span<const std::uint64_t>(counts));
    detail::put(out, std::span<const Point>(mesh.pos));
    static_assert(sizeof(Shape) == sizeof(std::uint8_t));
    detail::put(out, std::span<const Shape>(mesh.type));
    detail::put(out, std::span<const Index>(mesh.con));
    detail::put(out, std::span<const Index>(node));
    detail::put(out, std::span<const Index>(count));
    detail::put(out, std::span<const Index>(parent));
    detail::put(out, std::span<const double>(weight));
    const auto put_name = [&out](const std::string& name, std::uint8_t dtype, std::uint64_t n) {
        const auto len = static_cast<std::uint32_t>(name.size());
        detail::put(out, std::span<const std::uint32_t>(&len, 1));
        out.write(name.data(), static_cast<std::streamsize>(name.size()));
        detail::put(out, std::span<const std::uint8_t>(&dtype, 1));
        detail::put(out, std::span<const std::uint64_t>(&n, 1));
    };
    for (const auto& [name, values] : ints) {
        put_name(name, 0, values.size());
        detail::put(out, std::span<const Index>(values));
    }
    for (const auto& [name, values] : doubles) {
        put_name(name, 1, values.size());
        detail::put(out, std::span<const double>(values));
    }
    if (!out)
        throw std::runtime_error("SMX2: failed writing " + path);
}

inline Payload read(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("SMX2: cannot open " + path);
    std::uint64_t left = std::filesystem::file_size(path);

    const auto magic = detail::take<char>(in, 4, left, path);
    if (!std::equal(magic.begin(), magic.end(), kMagic.begin()))
        throw std::runtime_error("SMX2: " + path + " is not an SMX2 file");
    const auto dim = detail::take_one<std::uint32_t>(in, left, path);
    if (dim != 2 && dim != 3)
        throw std::runtime_error("SMX2: dim must be 2 or 3");
    const auto counts = detail::take<std::uint64_t>(in, 5, left, path);
    constexpr auto kMax = static_cast<std::uint64_t>(std::numeric_limits<Index>::max());
    for (std::size_t i = 0; i < 4; ++i)
        if (counts[i] > kMax)
            throw std::runtime_error("SMX2: counts exceed the index type");

    Payload p;
    p.mesh.dim = static_cast<int>(dim);
    p.mesh.pos = detail::take<Point>(in, counts[0], left, path);
    const auto type = detail::take<std::uint8_t>(in, counts[1], left, path);
    std::uint64_t n_con = 0;
    for (auto t : type) {
        if (!known_shape(t) || element_type(static_cast<Shape>(t)).dim != p.mesh.dim)
            throw std::runtime_error("SMX2: " + path + " has an element that is unknown or not " +
                                     std::to_string(dim) + "D");
        n_con += static_cast<std::uint64_t>(element_type(static_cast<Shape>(t)).vertices);
        if (n_con > kMax)
            throw std::runtime_error("SMX2: counts exceed the index type");
        p.mesh.type.push_back(static_cast<Shape>(t));
        p.mesh.offset.push_back(static_cast<Index>(n_con));
    }
    p.mesh.con = detail::take<Index>(in, n_con, left, path);
    const auto node = detail::take<Index>(in, counts[2], left, path);
    const auto count = detail::take<Index>(in, counts[2], left, path);
    const auto parent = detail::take<Index>(in, counts[3], left, path);
    const auto weight = detail::take<double>(in, counts[3], left, path);
    std::uint64_t used = 0;
    for (std::size_t r = 0; r < node.size(); ++r) {
        if (count[r] < 1 || static_cast<std::uint64_t>(count[r]) > counts[3] - used)
            throw std::runtime_error("SMX2: " + path + " has bad constraint counts");
        const auto b = static_cast<std::ptrdiff_t>(used), e = b + count[r];
        p.mesh.hn.push_back({node[r],
                             {parent.begin() + b, parent.begin() + e},
                             {weight.begin() + b, weight.begin() + e}});
        used += static_cast<std::uint64_t>(count[r]);
    }
    if (used != counts[3])
        throw std::runtime_error("SMX2: " + path + " has bad constraint counts");
    for (std::uint64_t f = 0; f < counts[4]; ++f) {
        const auto len = detail::take_one<std::uint32_t>(in, left, path);
        const auto chars = detail::take<char>(in, len, left, path);
        const std::string name(chars.begin(), chars.end());
        const auto dtype = detail::take_one<std::uint8_t>(in, left, path);
        const auto n = detail::take_one<std::uint64_t>(in, left, path);
        if (p.ints.contains(name) || p.doubles.contains(name))
            throw std::runtime_error("SMX2: " + path + " repeats the field \"" + name + "\"");
        if (dtype == 0)
            p.ints[name] = detail::take<Index>(in, n, left, path);
        else if (dtype == 1)
            p.doubles[name] = detail::take<double>(in, n, left, path);
        else
            throw std::runtime_error("SMX2: unknown field type in " + path);
    }
    if (left != 0)
        throw std::runtime_error("SMX2: " + path + " has trailing bytes");
    return p;
}

}  // namespace amr::simplex::io
