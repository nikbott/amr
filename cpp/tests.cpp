#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <random>
#include <unordered_map>
#include <set>
#include <iomanip>

#include "tree.hpp"
#include "morton.hpp"
#include "config.hpp"
#include "physics.hpp"

// ==================================================================================
// 1. SFC CORRECTNESS
// ==================================================================================

void test_morton_bijection() {
    std::cout << "[Test] SFC Bijection (Encode/Decode)... ";
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint32_t> dist(0, 1000000);
    
    // 2D
    for (int i = 0; i < 1000; ++i) {
        uint32_t x = dist(rng);
        uint32_t y = dist(rng);
        uint64_t code = Morton2D::encode(x, y);
        auto [dx, dy] = Morton2D::decode(code);
        assert(x == dx && y == dy);
    }
    // 3D
    for (int i = 0; i < 1000; ++i) {
        uint32_t x = dist(rng), y = dist(rng), z = dist(rng);
        uint64_t code = Morton3D::encode(x, y, z);
        auto [dx, dy, dz] = Morton3D::decode(code);
        assert(x == dx && y == dy && z == dz);
    }
    std::cout << "PASSED" << std::endl;
}

// ==================================================================================
// 2. LINEAR TREE COMPLETENESS
// ==================================================================================

template <int DIM>
void test_tree_completeness() {
    std::cout << "[Test] Linear Tree Completeness (" << DIM << "D)... ";
    AMRConfig config;
    config.max_level = 5; 
    LinearTree<DIM> tree(config.max_level);

    std::mt19937 rng(123);
    auto random_oracle = [&](const Node& n, int) { return (n.level < 4 && (rng() % 3 == 0)); };
    for(int i=0; i<5; ++i) tree.refine(random_oracle);

    // 1. Sorting Invariant (Crucial for Linear Octree)
    assert(std::is_sorted(tree.leaves.begin(), tree.leaves.end()) && "Leaves not sorted after refinement");

    // 2. Uniqueness Invariant
    auto it = std::adjacent_find(tree.leaves.begin(), tree.leaves.end());
    assert(it == tree.leaves.end() && "Duplicate Morton codes found");

    // 3. Volume Conservation Invariant
    double total_volume = 0.0;
    for (const auto& node : tree.leaves) {
        double side = 1.0 / (1ULL << node.level);
        total_volume += std::pow(side, DIM);
    }
    assert(std::abs(total_volume - 1.0) < 1e-9 && "Mesh volume error");
    
    std::cout << "PASSED" << std::endl;
}

// ==================================================================================
// 3. COARSENING & REVERSIBILITY
// ==================================================================================

template <int DIM>
void test_coarsening() {
    std::cout << "[Test] Refine/Coarsen Reversibility (" << DIM << "D)... ";
    
    AMRConfig config;
    config.max_level = 6;
    LinearTree<DIM> tree(config.max_level);

    // 1. Refine Uniformly to Level 3
    auto refine_oracle = [&](const Node& n, int) { return n.level < 3; };
    for(int i=0; i<10; ++i) if(!tree.refine(refine_oracle)) break;
    
    size_t size_refined = tree.leaves.size();
    size_t expected_size = 1ULL << (DIM * 3); 
    assert(size_refined == expected_size);

    // 2. Coarsen back to Level 2
    // Oracle returns FALSE to indicate "Do not refine" (i.e. collapse)
    auto coarsen_oracle = [&](const Node& n, int) { return n.level < 2; };

    bool changed = tree.coarsen(coarsen_oracle);
    
    // Verify Size
    size_t size_coarsened = tree.leaves.size();
    size_t expected_coarse = 1ULL << (DIM * 2);

    if (size_coarsened != expected_coarse) {
        std::cerr << "Coarsen failed. Expected " << expected_coarse << " got " << size_coarsened << std::endl;
        assert(false);
    }
    assert(changed == true);

    // Verify Sorting Invariant (Critical after removing std::sort)
    assert(std::is_sorted(tree.leaves.begin(), tree.leaves.end()) && "Leaves not sorted after coarsening");

    std::cout << "PASSED" << std::endl;
}

// ==================================================================================
// 4. IRREGULAR GEOMETRY & BALANCE
// ==================================================================================

void test_sphere_mesh_balance() {
    std::cout << "[Test] Sphere Mesh 2:1 Balance (3D)... ";
    
    AMRConfig config;
    config.max_level = 8;
    config.fine_level = 6;
    config.coarse_level = 2;
    config.radius = 0.4;
    config.center = {0.5, 0.5, 0.5};
    config.bandwidth = 0.05;

    LinearTree<3> tree(config.max_level);
    
    auto centers = config.get_int_center(3);
    SphereOracle3D oracle(
        centers[0], centers[1], centers[2],
        config.get_int_radius(),
        config.get_int_bandwidth(),
        config.coarse_level,
        config.fine_level
    );

    // 1. Refine
    for (int i = 0; i < 20; ++i) {
        if (!tree.refine(oracle)) break;
    }

    // 2. Balance
    tree.balance();

    // 3. Verify Balance
    int violations = 0;
    std::vector<std::vector<int>> dirs = {{1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1}};
    
    // Map for fast lookups (Exact match)
    std::unordered_map<uint64_t, int> level_map;
    for(const auto& n : tree.leaves) level_map[n.code] = n.level;

    for (const auto& node : tree.leaves) {
        auto [coords, size] = tree.get_geometry(node);
        
        for (const auto& dir : dirs) {
            // Find neighbor level geometrically to be robust against encoding
            std::vector<double> n_center(3);
            for(int k=0; k<3; ++k) n_center[k] = coords[k] + size/2.0 + dir[k] * size;

            if (n_center[0] < 0 || n_center[0] >= tree.domain_width() ||
                n_center[1] < 0 || n_center[1] >= tree.domain_width() ||
                n_center[2] < 0 || n_center[2] >= tree.domain_width()) continue;

            // Simple map check for same-level neighbor
            uint64_t n_code = tree.get_neighbor_code(node.code, node.level, dir);
            if (level_map.count(n_code)) {
                // Neighbor is same level (diff = 0) -> OK
                continue; 
            }

            // Fallback: Linear Search for coarser/finer neighbor
            // (Only done if exact match fails, to keep test time reasonable)
            int neighbor_lvl = -100;
            for(const auto& other : tree.leaves) {
                auto [o_coords, o_size] = tree.get_geometry(other);
                if (n_center[0] >= o_coords[0] && n_center[0] < o_coords[0] + o_size &&
                    n_center[1] >= o_coords[1] && n_center[1] < o_coords[1] + o_size &&
                    n_center[2] >= o_coords[2] && n_center[2] < o_coords[2] + o_size) {
                    neighbor_lvl = other.level;
                    break;
                }
            }

            if (neighbor_lvl != -100) {
                if (std::abs(node.level - neighbor_lvl) > 1) {
                    violations++;
                }
            }
        }
    }

    if (violations > 0) std::cout << "FAILED (" << violations << " violations)" << std::endl;
    else std::cout << "PASSED" << std::endl;
    
    assert(violations == 0);
}

void test_64bit_safety() {
    std::cout << "[Test] 64-bit Arithmetic Safety (Level 21)... ";
    uint64_t max_coord = (1ULL << 21) - 1;
    uint64_t code = Morton3D::encode(max_coord, max_coord, max_coord);
    assert(code > 0);
    auto [x, y, z] = Morton3D::decode(code);
    assert(x == max_coord && y == max_coord && z == max_coord);
    std::cout << "PASSED" << std::endl;
}

// ==================================================================================
// MAIN
// ==================================================================================

int main() {
    std::cout << "=== RUNNING COMPLIANCE TEST SUITE ===\n" << std::endl;
    test_morton_bijection();
    test_tree_completeness<2>();
    test_tree_completeness<3>();
    test_coarsening<2>();
    test_coarsening<3>();
    test_sphere_mesh_balance();
    test_64bit_safety();
    std::cout << "\n=== ALL TESTS PASSED ===" << std::endl;
    return 0;
}