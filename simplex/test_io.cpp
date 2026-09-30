/**
 * @file test_io.cpp
 * @brief SMX2 (io.hpp) must round-trip meshes and fields and reject bad files.
 */

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "io.hpp"
#include "test_util.hpp"

using namespace amr::simplex;
using namespace amr::simplex::test;

namespace {

std::filesystem::path temp_file(const std::string& name) {
    return std::filesystem::temp_directory_path() / ("simplex_" + name + ".smx");
}

/// Overwrites `bytes` at `offset` of an existing file.
void patch(const std::filesystem::path& path, std::streamoff offset, const std::string& bytes) {
    std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
    f.seekp(offset);
    f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

}  // namespace

TEST_CASE("SMX2 round-trips a mesh and its fields", "[simplex][io]") {
    const int dim = GENERATE(2, 3);
    const bool hybrid = GENERATE(false, true);
    std::mt19937 rng(41);
    const Mesh seed =
        hybrid ? (dim == 2 ? hybrid_2d(3, rng) : hybrid_3d(2, rng)) : grid(dim, 2, 0.2, rng);
    const Mesh m = randomly_refined(seed, 2, 0.3, rng);
    REQUIRE(!m.hn.empty());
    const std::map<std::string, std::vector<Index>> ints{{"elements", {3, 1, 4, 1, 5}},
                                                         {"empty", {}}};
    const std::map<std::string, std::vector<double>> doubles{{"error", {0.5, -1e300, 2.25}}};
    const auto path = temp_file("roundtrip_" + std::to_string(dim));

    io::write(path.string(), m, ints, doubles);
    const auto p = io::read(path.string());
    std::filesystem::remove(path);
    CHECK(p.mesh.dim == m.dim);
    CHECK(p.mesh.pos == m.pos);
    CHECK(p.mesh.type == m.type);
    CHECK(p.mesh.offset == m.offset);
    CHECK(p.mesh.con == m.con);
    CHECK(p.mesh.hn == m.hn);
    CHECK(p.ints == ints);
    CHECK(p.doubles == doubles);
}

TEST_CASE("SMX2 carries an empty mesh with fields", "[simplex][io]") {
    Mesh empty;
    empty.dim = 3;
    const auto path = temp_file("empty");
    io::write(path.string(), empty, {{"nodes", {4, 4, 4}}}, {{"dims", {1.0, 2.0, 3.0}}});
    const auto p = io::read(path.string());
    std::filesystem::remove(path);
    CHECK(p.mesh.dim == 3);
    CHECK(p.mesh.pos.empty());
    CHECK(p.mesh.num_elements() == 0);
    CHECK(p.ints.at("nodes") == std::vector<Index>{4, 4, 4});
}

TEST_CASE("SMX2 rejects foreign, truncated, padded and inconsistent files", "[simplex][io]") {
    std::mt19937 rng(43);
    const Mesh m = randomly_refined(grid(2, 2, 0.0, rng), 1, 0.3, rng);
    REQUIRE(!m.hn.empty());
    const auto path = temp_file("corrupt");
    const auto fresh = [&] { io::write(path.string(), m, {{"elements", {0}}}); };
    // Offsets: magic 4, dim 4, five counts 40, then pos, types, con, constraints.
    const std::streamoff types = 4 + 4 + 40 + static_cast<std::streamoff>(m.pos.size() * 24);
    const std::streamoff counts = types + m.num_elements() +
                                  static_cast<std::streamoff>(m.con.size() * 4) +
                                  static_cast<std::streamoff>(m.hn.size() * 4);

    fresh();
    std::filesystem::resize_file(path, std::filesystem::file_size(path) - 1);
    CHECK_THROWS_AS(io::read(path.string()), std::runtime_error);

    fresh();
    std::ofstream(path, std::ios::binary | std::ios::app).put('\0');
    CHECK_THROWS_AS(io::read(path.string()), std::runtime_error);  // trailing byte

    fresh();
    patch(path, 0, "X");
    CHECK_THROWS_AS(io::read(path.string()), std::runtime_error);  // magic

    for (const char code : {'\2', '\4'}) {  // unknown, and a 3D shape in a 2D file
        fresh();
        patch(path, types, std::string(1, code));
        CHECK_THROWS_AS(io::read(path.string()), std::runtime_error);
    }

    for (const Index count : {0, 3}) {  // a row without parents, rows overrunning the parents
        fresh();
        patch(path, counts, std::string(reinterpret_cast<const char*>(&count), sizeof count));
        CHECK_THROWS_AS(io::read(path.string()), std::runtime_error);
    }

    fresh();
    const std::uint64_t huge = std::uint64_t{1} << 40;
    patch(path, 8, std::string(reinterpret_cast<const char*>(&huge), sizeof huge));
    CHECK_THROWS_AS(io::read(path.string()), std::runtime_error);  // count beyond the file

    io::write(path.string(), m, {{"elements", {0}}}, {{"elements", {1.0}}});  // one name, twice
    CHECK_THROWS_AS(io::read(path.string()), std::runtime_error);

    std::filesystem::remove(path);
    CHECK_THROWS_AS(io::read(path.string()), std::runtime_error);
}
