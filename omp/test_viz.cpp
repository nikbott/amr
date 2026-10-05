/**
 * @file test_viz.cpp
 * @brief VTK output (common/viz.hpp).
 */

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "../common/viz.hpp"
#include "test_util.hpp"

using namespace amr;
using namespace amr::test;

TEST_CASE("viz::write_vtk emits UNSTRUCTURED_GRID cells, not a point cloud", "[viz]") {
    // Guards the single-source writer's core contract: one VTK cell per leaf with
    // the right cell type. The per-backend copies this replaced had diverged to a
    // VTK_VERTEX point cloud (one point per leaf) and a mislabelled VTK_VOXEL --
    // both silently wrong in ParaView. Build a LeafView by hand (no tree needed)
    // and read the file back.
    namespace fs = std::filesystem;
    auto slurp = [](const std::string& p) {
        std::ifstream f(p);
        std::stringstream ss;
        ss << f.rdbuf();
        return ss.str();
    };

    SECTION("2D -> VTK_QUAD") {
        std::vector<uint64_t> codes = {0, 1, 2, 3};  // four level-1 quads
        std::vector<uint8_t> levels = {1, 1, 1, 1};
        viz::LeafView v{codes.data(), levels.data(), 4, 1, 2};
        const std::string path = (fs::temp_directory_path() / "amr_viz_2d.vtk").string();
        viz::write_vtk(path, v);
        const std::string s = slurp(path);
        fs::remove(path);
        REQUIRE(s.find("DATASET UNSTRUCTURED_GRID") != std::string::npos);
        REQUIRE(s.find("POINTS 16 double") != std::string::npos);  // 4 quads x 4 corners
        REQUIRE(s.find("CELLS 4 20") != std::string::npos);        // n, n*(4+1)
        REQUIRE(s.find("CELL_TYPES 4\n9") != std::string::npos);   // 9 = VTK_QUAD
    }

    SECTION("3D -> VTK_HEXAHEDRON") {
        std::vector<uint64_t> codes = {0, 1, 2, 3, 4, 5, 6, 7};  // eight level-1 hexes
        std::vector<uint8_t> levels = {1, 1, 1, 1, 1, 1, 1, 1};
        viz::LeafView v{codes.data(), levels.data(), 8, 1, 3};
        const std::string path = (fs::temp_directory_path() / "amr_viz_3d.vtk").string();
        viz::write_vtk(path, v);
        const std::string s = slurp(path);
        fs::remove(path);
        REQUIRE(s.find("DATASET UNSTRUCTURED_GRID") != std::string::npos);
        REQUIRE(s.find("POINTS 64 double") != std::string::npos);  // 8 hexes x 8 corners
        REQUIRE(s.find("CELLS 8 72") != std::string::npos);        // n, n*(8+1)
        REQUIRE(s.find("CELL_TYPES 8\n12") != std::string::npos);  // 12 = VTK_HEXAHEDRON
    }
}
