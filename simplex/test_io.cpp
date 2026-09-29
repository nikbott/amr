/**
 * @file test_io.cpp
 * @brief SMX1 (io.hpp) must round-trip meshes and reject bad files.
 */

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

TEST_CASE("SMX1 round-trips a mesh and its indices", "[simplex][io]") {
    const int dim = GENERATE(2, 3);
    std::mt19937 rng(41);
    const Mesh m = randomly_refined(grid(dim, 2, 0.2, rng), 2, 0.3, rng);
    REQUIRE(!m.hn.empty());
    const std::vector<Index> indices{3, 1, 4, 1, 5};
    const auto path = std::filesystem::temp_directory_path() /
                      ("simplex_roundtrip_" + std::to_string(dim) + ".smx");

    io::write(path.string(), m, indices);
    const auto p = io::read(path.string());
    std::filesystem::remove(path);
    CHECK(p.mesh.dim == m.dim);
    CHECK(p.mesh.pos == m.pos);
    CHECK(p.mesh.con == m.con);
    CHECK(p.mesh.hn == m.hn);
    CHECK(p.indices == indices);
}

TEST_CASE("SMX1 rejects foreign and truncated files", "[simplex][io]") {
    std::mt19937 rng(43);
    const Mesh m = grid(2, 2, 0.0, rng);
    const auto path = std::filesystem::temp_directory_path() / "simplex_corrupt.smx";
    io::write(path.string(), m, std::vector<Index>{0});
    const auto size = std::filesystem::file_size(path);

    std::filesystem::resize_file(path, size - 1);
    CHECK_THROWS_AS(io::read(path.string()), std::runtime_error);

    io::write(path.string(), m, std::vector<Index>{0});
    {
        std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
        f.put('X');
    }
    CHECK_THROWS_AS(io::read(path.string()), std::runtime_error);
    std::filesystem::remove(path);
    CHECK_THROWS_AS(io::read(path.string()), std::runtime_error);
}
