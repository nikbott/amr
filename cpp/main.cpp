#include "config.hpp"
#include "tree.hpp"
#include "physics.hpp"
#include "viz.hpp"
#include <chrono>
#include <iostream>

int main() {    
    // ==========================================
    // 2D DEMO (Quadtree)
    // ==========================================
    std::cout << "\n--- 2D QUADTREE DEMO ---" << std::endl;

    AMRConfig cfg;
    cfg.max_level = 16;
    cfg.fine_level = 8;
    cfg.coarse_level = 3;

    Quadtree tree_2d(cfg.max_level);
    
    auto centers = cfg.get_int_center(2);
    CircleOracle2D oracle(
        centers[0], 
        centers[1], 
        cfg.get_int_radius(), 
        cfg.get_int_bandwidth(), 
        cfg.coarse_level, 
        cfg.fine_level
    );

    std::cout << "Refining mesh (Circle)..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 10; ++i) {
        if (!tree_2d.refine(oracle)) break;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Refinement Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    std::cout << "Elements before balancing: " << tree_2d.leaves.size() << std::endl;

    std::cout << "Applying 2:1 Ripple Balance..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    tree_2d.balance();
    end = std::chrono::high_resolution_clock::now();
    
    std::cout << "Balance Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    std::cout << "Elements after balancing:  " << tree_2d.leaves.size() << std::endl;

    MeshVisualizer::save_svg(tree_2d, "mesh_2d.svg");

    // ==========================================
    // 3D DEMO (Octree)
    // ==========================================
    std::cout << "\n--- 3D OCTREE DEMO ---" << std::endl;

    Octree tree_3d(cfg.max_level);
    auto centers_3d = cfg.get_int_center(3);
    
    SphereOracle3D oracle_3d(
        centers_3d[0], centers_3d[1], centers_3d[2],
        cfg.get_int_radius(),
        cfg.get_int_bandwidth(),
        cfg.coarse_level,
        cfg.fine_level
    );

    std::cout << "Refining mesh (Sphere)..." << std::endl;
    start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 10; ++i) {
        if (!tree_3d.refine(oracle_3d)) break;
    }
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Refinement Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    std::cout << "Elements before balancing: " << tree_3d.leaves.size() << std::endl;

    std::cout << "Applying 2:1 Ripple Balance..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    tree_3d.balance();
    end = std::chrono::high_resolution_clock::now();
    
    std::cout << "Balance Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    std::cout << "Elements after balancing:  " << tree_3d.leaves.size() << std::endl;

    MeshVisualizer::save_obj(tree_3d, "mesh_3d.obj");

    return 0;
}