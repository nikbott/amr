/**
 * @file test_mesh_io.cpp
 * @brief The AMR1 mesh+field format (common/mesh_io.hpp).
 */

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "../common/mesh_io.hpp"
#include "test_util.hpp"

using namespace amr;
using namespace amr::test;
using namespace Catch::Matchers;

TEMPLATE_TEST_CASE("Binary mesh+field format round-trip (mesh_io C.2)",
                   "[mesh_io][integration]",
                   Quadtree,
                   Octree) {
    constexpr int DIM = (std::is_same<TestType, Quadtree>::value) ? 2 : 3;

    // Build a non-uniform mesh.
    int max_lvl = 6;
    TestType tree(max_lvl);
    std::mt19937_64 rng(7);
    for (int i = 0; i < 4; ++i)
        tree.refine([&](const Node& n, int) { return n.level < 5 && (rng() % 3 == 0); });

    // Pack into a MeshData with a non-trivial bbox + two fields (f64 and f32).
    mesh_io::MeshData m;
    m.dim = DIM;
    m.max_level = static_cast<uint32_t>(max_lvl);
    m.origin = {{1.5, -2.0, 3.25}};
    m.size = {{100.0, 50.0, 12.5}};
    for (const auto& node : tree) {
        m.codes.push_back(node.code.value);
        m.levels.push_back(static_cast<uint8_t>(node.level));
    }
    mesh_io::Field err{"dic_error", true, {}};
    mesh_io::Field lvl{"level_f32", false, {}};
    for (size_t i = 0; i < m.codes.size(); ++i) {
        err.values.push_back(std::sin(static_cast<double>(i)) * 1.0e-3);
        lvl.values.push_back(static_cast<double>(m.levels[i]));
    }
    m.fields = {err, lvl};

    auto path =
        (std::filesystem::temp_directory_path() / ("amr_mesh_io_" + std::to_string(DIM) + "d.bin"))
            .string();

    SECTION("write/read preserves geometry, codes, levels, and fields") {
        mesh_io::write(path, m);
        mesh_io::MeshData r = mesh_io::read(path);

        REQUIRE(r.dim == m.dim);
        REQUIRE(r.max_level == m.max_level);
        for (uint32_t k = 0; k < m.dim; ++k) {
            REQUIRE(r.origin[k] == m.origin[k]);  // f64 bbox is exact
            REQUIRE(r.size[k] == m.size[k]);
        }
        REQUIRE(r.codes == m.codes);  // bit-exact
        REQUIRE(r.levels == m.levels);
        REQUIRE(r.elem_type == mesh_io::default_elem_type(DIM));  // unset -> dim default

        REQUIRE(r.fields.size() == 2);
        REQUIRE(r.fields[0].name == "dic_error");
        REQUIRE(r.fields[0].f64);
        REQUIRE(r.fields[0].values == err.values);  // f64 exact
        REQUIRE(r.fields[1].name == "level_f32");
        REQUIRE_FALSE(r.fields[1].f64);
        for (size_t i = 0; i < lvl.values.size(); ++i)
            REQUIRE_THAT(r.fields[1].values[i], WithinAbs(lvl.values[i], 1e-5));  // f32 round-off
        std::filesystem::remove(path);
    }

    SECTION("rejects a corrupt magic") {
        {
            std::ofstream bad(path, std::ios::binary);
            bad << "XXXXnonsense";
        }
        REQUIRE_THROWS_AS(mesh_io::read(path), std::runtime_error);
        std::filesystem::remove(path);
    }
}

TEST_CASE("Binary mesh+field format edge cases (mesh_io C.2)", "[mesh_io]") {
    auto path = (std::filesystem::temp_directory_path() / "amr_mesh_io_edge.bin").string();

    SECTION("empty mesh and field-less round-trip") {
        mesh_io::MeshData m;
        m.dim = 3;
        m.max_level = 10;
        mesh_io::write(path, m);
        mesh_io::MeshData r = mesh_io::read(path);
        REQUIRE(r.codes.empty());
        REQUIRE(r.fields.empty());
        REQUIRE(r.dim == 3);
        std::filesystem::remove(path);
    }

    SECTION("padding: n not a multiple of 8 round-trips with a trailing field") {
        mesh_io::MeshData m;
        m.dim = 2;
        m.max_level = 4;
        m.codes = {0, 1, 2, 3, 4};  // n = 5 -> 3 pad bytes before n_fields
        m.levels = {1, 1, 1, 1, 1};
        m.fields = {mesh_io::Field{"f", true, {0.1, 0.2, 0.3, 0.4, 0.5}}};
        mesh_io::write(path, m);
        mesh_io::MeshData r = mesh_io::read(path);
        REQUIRE(r.codes == m.codes);
        REQUIRE(r.fields.size() == 1);
        REQUIRE(r.fields[0].values == m.fields[0].values);
        std::filesystem::remove(path);
    }
}

TEST_CASE("AMR1 element-type tag (mesh_io #12)", "[mesh_io][elem_type]") {
    namespace mio = amr::mesh_io;
    auto path = (std::filesystem::temp_directory_path() / "amr_mesh_io_etype.bin").string();

    SECTION("unset tag defaults to the dim simplex on write") {
        for (uint32_t dim : {2u, 3u}) {
            mio::MeshData m;
            m.dim = dim;
            mio::write(path, m);
            mio::MeshData r = mio::read(path);
            REQUIRE(r.elem_type == mio::default_elem_type(dim));
        }
        std::filesystem::remove(path);
    }

    SECTION("an explicit tag round-trips") {
        mio::MeshData m;
        m.dim = 2;
        m.elem_type = mio::ElemType::Q4;  // overriding the T3 default
        mio::write(path, m);
        mio::MeshData r = mio::read(path);
        REQUIRE(r.elem_type == mio::ElemType::Q4);
        std::filesystem::remove(path);
    }

    SECTION("a version-1 file (no tag) reads back as the dim default") {
        // Hand-write a minimal v1 header (empty mesh, no fields).
        {
            std::ofstream os(path, std::ios::binary | std::ios::trunc);
            os.write("AMR1", 4);
            mio::detail::put<uint32_t>(os, 1u);  // version 1: no elem_type on disk
            mio::detail::put<uint32_t>(os, 3u);  // dim
            mio::detail::put<uint32_t>(os, 5u);  // max_level
            mio::detail::put<uint64_t>(os, 0u);  // n_leaves
            for (int k = 0; k < 3; ++k)
                mio::detail::put<double>(os, 0.0);  // origin
            for (int k = 0; k < 3; ++k)
                mio::detail::put<double>(os, 1.0);  // size
            mio::detail::put<uint32_t>(os, 0u);     // n_fields
        }
        mio::MeshData r = mio::read(path);
        REQUIRE(r.elem_type == mio::ElemType::T4);  // dim==3 default
        std::filesystem::remove(path);
    }

    SECTION("an unknown element code is rejected") {
        {
            std::ofstream os(path, std::ios::binary | std::ios::trunc);
            os.write("AMR1", 4);
            mio::detail::put<uint32_t>(os, 2u);   // version 2
            mio::detail::put<uint32_t>(os, 2u);   // dim
            mio::detail::put<uint32_t>(os, 1u);   // max_level
            mio::detail::put<uint32_t>(os, 99u);  // bogus elem_type
            mio::detail::put<uint64_t>(os, 0u);   // n_leaves
            for (int k = 0; k < 2; ++k)
                mio::detail::put<double>(os, 0.0);
            for (int k = 0; k < 2; ++k)
                mio::detail::put<double>(os, 1.0);
            mio::detail::put<uint32_t>(os, 0u);
        }
        REQUIRE_THROWS_AS(mio::read(path), std::runtime_error);
        std::filesystem::remove(path);
    }
}

TEST_CASE("AMR1 reader rejects malformed input cleanly (mesh_io)", "[mesh_io][robustness]") {
    namespace mio = amr::mesh_io;
    auto path = (std::filesystem::temp_directory_path() / "amr_mesh_io_bad.bin").string();

    SECTION("an unknown field dtype is rejected, not silently read as float32") {
        {
            std::ofstream os(path, std::ios::binary | std::ios::trunc);
            os.write("AMR1", 4);
            mio::detail::put<uint32_t>(os, 2u);  // version 2
            mio::detail::put<uint32_t>(os, 2u);  // dim
            mio::detail::put<uint32_t>(os, 1u);  // max_level
            mio::detail::put<uint32_t>(os, 0u);  // elem_type T3
            mio::detail::put<uint64_t>(os, 0u);  // n_leaves = 0
            for (int k = 0; k < 2; ++k)
                mio::detail::put<double>(os, 0.0);
            for (int k = 0; k < 2; ++k)
                mio::detail::put<double>(os, 1.0);
            mio::detail::put<uint32_t>(os, 1u);  // n_fields = 1
            mio::detail::put<uint32_t>(os, 1u);  // name_len = 1
            os.put('f');                         // name
            mio::detail::put<uint32_t>(os, 2u);  // bogus dtype (only 0/1 valid)
        }
        REQUIRE_THROWS_AS(mio::read(path), std::runtime_error);
        std::filesystem::remove(path);
    }

    SECTION("a hostile leaf count throws runtime_error, not bad_alloc") {
        {
            std::ofstream os(path, std::ios::binary | std::ios::trunc);
            os.write("AMR1", 4);
            mio::detail::put<uint32_t>(os, 2u);     // version 2
            mio::detail::put<uint32_t>(os, 2u);     // dim
            mio::detail::put<uint32_t>(os, 1u);     // max_level
            mio::detail::put<uint32_t>(os, 0u);     // elem_type
            mio::detail::put<uint64_t>(os, ~0ull);  // n_leaves = 2^64-1 (hostile)
            for (int k = 0; k < 2; ++k)
                mio::detail::put<double>(os, 0.0);
            for (int k = 0; k < 2; ++k)
                mio::detail::put<double>(os, 1.0);
        }
        REQUIRE_THROWS_AS(mio::read(path), std::runtime_error);  // bounded before allocation
        std::filesystem::remove(path);
    }
}

// Snapshot the current leaf set in Morton (iteration) order -- the order the DIC
// error array is expected to align with.
