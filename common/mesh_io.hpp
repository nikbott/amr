/**
 * @file mesh_io.hpp
 * @brief Canonical binary mesh+field exchange format for the AMR <-> DIC bridge.
 *
 * @details Single little-endian file, the debuggable artifact handed between the
 * AMR engine and the MATLAB FE-DIC solver (plan Part C.2). Backend-agnostic:
 * operates on plain (codes, levels, bbox, fields) so omp/mpi/cuda can all emit
 * it. The MATLAB reader is `code/+mesh/amrIO.m`.
 *
 * Layout:
 *   [ magic    : 4 bytes  = "AMR1" ]
 *   [ version  : uint32   = 1 ]
 *   [ dim      : uint32   (2 or 3) ]
 *   [ max_level: uint32 ]
 *   [ n_leaves : uint64 ]
 *   [ bbox     : 2*dim float64 (origin[dim], then size[dim]) ]
 *   [ codes    : uint64 * n_leaves ]
 *   [ levels   : uint8  * n_leaves ]
 *   [ pad to 8-byte boundary ]
 *   [ n_fields : uint32 ]
 *     per field:
 *       [ name_len : uint32 ][ name : name_len bytes (no NUL) ]
 *       [ dtype : uint32 (0 = float32, 1 = float64) ]
 *       [ values: dtype * n_leaves ]
 *
 * Endianness: written in host byte order, which is little-endian on the x86/ARM
 * targets this runs on; the format is defined as little-endian. (A byte-swap
 * shim can be added if a big-endian consumer ever appears.)
 */
#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace amr::mesh_io {

// A per-leaf scalar field (e.g. a DIC error metric driving the oracle).
// Values are held as double; `f64` selects the on-disk precision.
struct Field {
    std::string name;
    bool f64 = true;
    std::vector<double> values;  // length == n_leaves
};

struct MeshData {
    uint32_t dim = 2;
    uint32_t max_level = 0;
    std::array<double, 3> origin{{0, 0, 0}};  // first `dim` used
    std::array<double, 3> size{{1, 1, 1}};    // first `dim` used
    std::vector<uint64_t> codes;
    std::vector<uint8_t> levels;
    std::vector<Field> fields;
};

namespace detail {

template <typename T>
inline void put(std::ostream& os, const T& v) {
    static_assert(std::is_trivially_copyable_v<T>, "raw POD write only");
    os.write(reinterpret_cast<const char*>(&v), static_cast<std::streamsize>(sizeof(T)));
}

template <typename T>
inline T get(std::istream& is) {
    static_assert(std::is_trivially_copyable_v<T>, "raw POD read only");
    T v{};
    is.read(reinterpret_cast<char*>(&v), static_cast<std::streamsize>(sizeof(T)));
    if (!is)
        throw std::runtime_error("mesh_io: unexpected end of file");
    return v;
}

inline uint64_t pad_to_8(uint64_t n) {
    return (8u - (n % 8u)) % 8u;
}

}  // namespace detail

/// Write `m` to `path`. Throws std::runtime_error on I/O failure or bad input.
inline void write(const std::string& path, const MeshData& m) {
    if (m.dim != 2 && m.dim != 3)
        throw std::runtime_error("mesh_io::write: dim must be 2 or 3");
    if (m.codes.size() != m.levels.size())
        throw std::runtime_error("mesh_io::write: codes/levels size mismatch");
    const uint64_t n = m.codes.size();
    for (const auto& f : m.fields)
        if (f.values.size() != n)
            throw std::runtime_error("mesh_io::write: field '" + f.name + "' wrong length");

    std::ofstream os(path, std::ios::binary | std::ios::trunc);
    if (!os)
        throw std::runtime_error("mesh_io::write: cannot open " + path);

    os.write("AMR1", 4);
    detail::put<uint32_t>(os, 1u);  // version
    detail::put<uint32_t>(os, m.dim);
    detail::put<uint32_t>(os, m.max_level);
    detail::put<uint64_t>(os, n);
    for (uint32_t k = 0; k < m.dim; ++k)
        detail::put<double>(os, m.origin[k]);
    for (uint32_t k = 0; k < m.dim; ++k)
        detail::put<double>(os, m.size[k]);

    if (n)
        os.write(reinterpret_cast<const char*>(m.codes.data()),
                 static_cast<std::streamsize>(n * sizeof(uint64_t)));
    if (n)
        os.write(reinterpret_cast<const char*>(m.levels.data()), static_cast<std::streamsize>(n));
    const uint64_t pad = detail::pad_to_8(n);
    for (uint64_t i = 0; i < pad; ++i)
        os.put('\0');

    detail::put<uint32_t>(os, static_cast<uint32_t>(m.fields.size()));
    for (const auto& f : m.fields) {
        detail::put<uint32_t>(os, static_cast<uint32_t>(f.name.size()));
        os.write(f.name.data(), static_cast<std::streamsize>(f.name.size()));
        detail::put<uint32_t>(os, f.f64 ? 1u : 0u);
        if (f.f64) {
            os.write(reinterpret_cast<const char*>(f.values.data()),
                     static_cast<std::streamsize>(n * sizeof(double)));
        } else {
            for (uint64_t i = 0; i < n; ++i)
                detail::put<float>(os, static_cast<float>(f.values[i]));
        }
    }
    if (!os)
        throw std::runtime_error("mesh_io::write: write failed for " + path);
}

/// Read a MeshData from `path`. Throws std::runtime_error on bad magic/version/EOF.
inline MeshData read(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is)
        throw std::runtime_error("mesh_io::read: cannot open " + path);

    char magic[4];
    is.read(magic, 4);
    if (!is || std::memcmp(magic, "AMR1", 4) != 0)
        throw std::runtime_error("mesh_io::read: bad magic (not an AMR1 file)");
    const uint32_t version = detail::get<uint32_t>(is);
    if (version != 1)
        throw std::runtime_error("mesh_io::read: unsupported version");

    MeshData m;
    m.dim = detail::get<uint32_t>(is);
    if (m.dim != 2 && m.dim != 3)
        throw std::runtime_error("mesh_io::read: bad dim");
    m.max_level = detail::get<uint32_t>(is);
    const uint64_t n = detail::get<uint64_t>(is);
    for (uint32_t k = 0; k < m.dim; ++k)
        m.origin[k] = detail::get<double>(is);
    for (uint32_t k = 0; k < m.dim; ++k)
        m.size[k] = detail::get<double>(is);

    m.codes.resize(n);
    m.levels.resize(n);
    if (n)
        is.read(reinterpret_cast<char*>(m.codes.data()),
                static_cast<std::streamsize>(n * sizeof(uint64_t)));
    if (n)
        is.read(reinterpret_cast<char*>(m.levels.data()), static_cast<std::streamsize>(n));
    if (!is)
        throw std::runtime_error("mesh_io::read: truncated codes/levels");
    is.seekg(static_cast<std::streamoff>(detail::pad_to_8(n)), std::ios::cur);

    const uint32_t n_fields = detail::get<uint32_t>(is);
    m.fields.reserve(n_fields);
    for (uint32_t fi = 0; fi < n_fields; ++fi) {
        Field f;
        const uint32_t name_len = detail::get<uint32_t>(is);
        f.name.resize(name_len);
        if (name_len)
            is.read(f.name.data(), static_cast<std::streamsize>(name_len));
        const uint32_t dtype = detail::get<uint32_t>(is);
        f.f64 = (dtype == 1);
        f.values.resize(n);
        if (f.f64) {
            if (n)
                is.read(reinterpret_cast<char*>(f.values.data()),
                        static_cast<std::streamsize>(n * sizeof(double)));
        } else {
            for (uint64_t i = 0; i < n; ++i)
                f.values[i] = static_cast<double>(detail::get<float>(is));
        }
        if (!is)
            throw std::runtime_error("mesh_io::read: truncated field '" + f.name + "'");
        m.fields.push_back(std::move(f));
    }
    return m;
}

}  // namespace amr::mesh_io
