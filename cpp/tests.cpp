#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <random>
#include <unordered_map>
#include <iomanip>
#include <array>

#include "tree.hpp"
#include "core.hpp"
#include "physics.hpp"

using namespace amr;

// ==================================================================================
// 1. SFC CORRECTNESS
// ==================================================================================

void test_morton_bijection() {
    std::cout << "[Test] SFC Bijection (Encode/Decode)... ";
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint32_t> dist(0, 1000000);
    
    // 2D Test
    for (int i = 0; i < 10000; ++i) {
        uint32_t x = dist(rng);
        uint32_t y = dist(rng);
        uint64_t code = morton::encode_2d(x, y);
        auto [dx, dy] = morton::decode_2d(code);
        assert(x == dx && y == dy);
    }

    // 3D Test
    for (int i = 0; i < 10000; ++i) {
        uint32_t x = dist(rng), y = dist(rng), z = dist(rng);
        uint64_t code = morton::encode_3d(x, y, z);
        auto [dx, dy, dz] = morton::decode_3d(code);
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
    
    // Setup a small tree
    int max_lvl = 5; 
    LinearTree<DIM> tree(max_lvl);

    // Refine randomly
    std::mt19937 rng(123);
    auto random_oracle = [&](const Node& n, int) { 
        return (n.level < 4 && (rng() % 3 == 0)); 
    };
    
    for(int i=0; i<5; ++i) tree.refine(random_oracle);

    // 1. Sorting Invariant (Crucial for Burstedde's Linear Octree)
    assert(std::is_sorted(tree.leaves.begin(), tree.leaves.end()) && "Leaves not sorted by Morton code");

    // 2. Uniqueness Invariant
    auto it = std::adjacent_find(tree.leaves.begin(), tree.leaves.end());
    assert(it == tree.leaves.end() && "Duplicate Morton codes found");

    // 3. Volume Conservation Invariant (Sum of volumes == Domain Volume)
    double total_volume = 0.0;
    for (const auto& node : tree.leaves) {
        // Size relative to domain (1.0)
        double side = 1.0 / (1ULL << node.level);
        total_volume += std::pow(side, DIM);
    }
    
    // Epsilon check for floating point
    assert(std::abs(total_volume - 1.0) < 1e-9 && "Mesh volume error (Conservation violation)");
    
    std::cout << "PASSED" << std::endl;
}

// ==================================================================================
// 3. COARSENING & REVERSIBILITY
// ==================================================================================

template <int DIM>
void test_coarsening() {
    std::cout << "[Test] Refine/Coarsen Reversibility (" << DIM << "D)... ";
    
    int max_lvl = 6;
    LinearTree<DIM> tree(max_lvl);

    // 1. Refine Uniformly to Level 3
    auto refine_oracle = [&](const Node& n, int) { return n.level < 3; };
    while(tree.refine(refine_oracle));
    
    size_t size_refined = tree.leaves.size();
    size_t expected_size = 1ULL << (DIM * 3); 
    assert(size_refined == expected_size);

    // 2. Coarsen back to Level 2
    auto coarsen_oracle = [&](const Node& n, int) { return n.level < 2; };

    bool changed = tree.coarsen(coarsen_oracle);
    
    size_t size_coarsened = tree.leaves.size();
    size_t expected_coarse = 1ULL << (DIM * 2);

    assert(changed == true);
    assert(size_coarsened == expected_coarse);
    assert(std::is_sorted(tree.leaves.begin(), tree.leaves.end()));

    std::cout << "PASSED" << std::endl;
}

// ==================================================================================
// 4. BALANCE VERIFICATION
// ==================================================================================

template<int DIM>
void verify_balance(const LinearTree<DIM>& tree) {
    std::unordered_map<uint64_t, int> level_map;
    level_map.reserve(tree.leaves.size());
    for(const auto& n : tree.leaves) level_map[n.code] = n.level;

    std::vector<std::array<int, DIM>> dirs;
    if constexpr (DIM == 2) dirs = {{1,0}, {-1,0}, {0,1}, {0,-1}};
    else dirs = {{1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1}};

    int violations = 0;
    
    for (const auto& node : tree.leaves) {
        auto coords = tree.decode(node.code);
        uint64_t size = 1ULL << (tree.max_level - node.level);

        for (const auto& dir : dirs) {
            std::vector<double> n_center(DIM);
            bool boundary = false;
            
            for(int k=0; k<DIM; ++k) {
                double c = coords[k] + size/2.0 + dir[k] * size;
                if (c < 0 || c >= tree.domain_width()) boundary = true;
                n_center[k] = c;
            }
            if(boundary) continue;

            uint64_t n_code = tree.get_neighbor_code(node.code, node.level, dir);
            if (level_map.count(n_code)) continue;

            int neighbor_lvl = -100;
            // Linear search fallback for verification
            for(const auto& other : tree.leaves) {
                auto o_coords = tree.decode(other.code);
                uint64_t o_size = 1ULL << (tree.max_level - other.level);
                
                bool match = true;
                for(int k=0; k<DIM; ++k) {
                    if (n_center[k] < o_coords[k] || n_center[k] >= o_coords[k] + o_size) {
                        match = false; 
                        break;
                    }
                }
                if (match) {
                    neighbor_lvl = other.level;
                    break;
                }
            }

            if (neighbor_lvl != -100) {
                if (std::abs(node.level - neighbor_lvl) > 1) {
                    std::cout << "Violation: Node L" << node.level << " <-> Neighbor L" << neighbor_lvl << "\n";
                    violations++;
                }
            }
        }
    }
    assert(violations == 0 && "2:1 Balance Constraint Violated");
}

void test_sphere_mesh_balance() {
    std::cout << "[Test] Sphere Mesh 2:1 Balance (3D)... ";
    
    Config cfg;
    cfg.max_level = 8;
    // Set parameters to create a distinct shell
    cfg.coarse_level = 2;
    cfg.fine_level = 6;
    cfg.radius = 0.25;
    cfg.bandwidth = 0.05;
    
    Octree tree(cfg.max_level);
    SphereOracle oracle(cfg);

    // Refine until convergence
    for (int i = 0; i < 20; ++i) {
        if (!tree.refine(oracle)) break;
    }
    
    // Balance
    tree.balance();
    
    // Verify
    verify_balance(tree);
    
    std::cout << "PASSED" << std::endl;
}

void test_deep_ripple() {
    std::cout << "[Test] Deep Ripple (Point Refinement)... ";
    int max_lvl = 10;
    Quadtree tree(max_lvl);
    
    auto center_oracle = [&](const Node& n, int) {
        auto coords = tree.decode(n.code);
        uint64_t size = 1ULL << (max_lvl - n.level);
        uint64_t mid = tree.domain_width() / 2;
        
        bool contains_center = (coords[0] <= mid && coords[0] + size > mid && 
                                coords[1] <= mid && coords[1] + size > mid);
        
        return (contains_center && n.level < max_lvl);
    };

    while(tree.refine(center_oracle));
    tree.balance();
    verify_balance(tree);
    std::cout << "PASSED" << std::endl;
}

void test_random_cloud_balance() {
    std::cout << "[Test] Random Cloud Balance (3D)... ";
    int max_lvl = 8;
    Octree tree(max_lvl);
    
    auto cloud_oracle = [&](const Node& n, int) {
        if(n.level >= 6) return false;
        return (n.code % 7 == 0 || n.code % 13 == 0); 
    };

    for(int i=0; i<8; ++i) tree.refine(cloud_oracle);
    
    tree.balance();
    verify_balance(tree);
    
    std::cout << "PASSED" << std::endl;
}

int main() {
    std::cout << "=== RUNNING AMR TEST SUITE ===\n" << std::endl;
    
    test_morton_bijection();
    
    test_tree_completeness<2>();
    test_tree_completeness<3>();
    
    test_coarsening<2>();
    test_coarsening<3>();
    
    test_sphere_mesh_balance();
    test_deep_ripple();
    test_random_cloud_balance();
    
    std::cout << "\n=== ALL TESTS PASSED ===" << std::endl;
    return 0;
}