#include "tree.hpp"
#include "physics.hpp"
#include "viz.hpp"
#include <iostream>
#include <chrono>

using namespace amr;

int main() {
    std::cout << "=== AMR System ===\n";

    // 1. Configure
    Config cfg;
    cfg.max_level = 20;
    cfg.coarse_level = 3;
    cfg.fine_level = 9;
    cfg.radius = 0.35;

    // 2. 2D Simulation
    {
        std::cout << "\n--- 2D Quadtree ---\n";
        Quadtree tree(cfg.max_level);
        CircleOracle oracle(cfg);

        // Refine loop
        int steps = 0;
        while(tree.refine(oracle)) {
            std::cout << "Refine step " << ++steps << ": " << tree.leaves.size() << " leaves\n";
        }

        // Balance
        std::cout << "Balancing...\n";
        auto start = std::chrono::high_resolution_clock::now();
        tree.balance();
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Balanced in " 
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() 
                  << "ms. Final size: " << tree.leaves.size() << "\n";

        viz::write_svg(tree, "mesh_2d.svg");
    }

    // 3. 3D Simulation
    {
        std::cout << "\n--- 3D Octree ---\n";
        Octree tree(cfg.max_level);
        SphereOracle oracle(cfg);

        // Refine
        int steps = 0;
        while(tree.refine(oracle)) {
            std::cout << "Refine step " << ++steps << ": " << tree.leaves.size() << " leaves\n";
        }

        // Balance
        std::cout << "Balancing...\n";
        auto start = std::chrono::high_resolution_clock::now();
        tree.balance();
        auto end = std::chrono::high_resolution_clock::now();
        
        std::cout << "Balanced in " 
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() 
                  << "ms. Final size: " << tree.leaves.size() << "\n";

        viz::write_vtk(tree, "mesh_3d.vtk");
    }

    return 0;
}