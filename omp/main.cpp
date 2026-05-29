#include "tree.hpp"
#include "physics.hpp"
#include "viz.hpp"
#include <iostream>
#include <chrono>

using namespace amr;

int main() {
    std::cout << "=== AMR System (SoA + Strong Types) ===\n";

    Config cfg;
    cfg.max_level = 20;
    cfg.coarse_level = 3;
    cfg.fine_level = 10;
    cfg.radius = 0.25;

    // 2D Simulation
    {
        std::cout << "\n--- 2D Quadtree ---\n";
        Quadtree tree(cfg.max_level);
        CircleOracle oracle(cfg);

        int steps = 0;
        while(tree.refine(oracle)) {
            // New API usage: tree.size()
            std::cout << "Refine step " << ++steps << ": " << tree.size() << " leaves\n";
        }

        std::cout << "Balancing...\n";
        auto start = std::chrono::high_resolution_clock::now();
        tree.balance();
        auto end = std::chrono::high_resolution_clock::now();
        
        // Best Practice #4: Run Design-by-Contract verification
        tree.verify();

        std::cout << "Balanced in " 
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() 
                  << "ms. Final size: " << tree.size() << "\n";

        viz::write_svg(tree, "mesh_2d.svg");
    }

    // 3D Simulation
    {
        std::cout << "\n--- 3D Octree ---\n";
        Octree tree(cfg.max_level);
        SphereOracle oracle(cfg);

        int steps = 0;
        while(tree.refine(oracle)) {
            std::cout << "Refine step " << ++steps << ": " << tree.size() << " leaves\n";
        }

        std::cout << "Balancing...\n";
        auto start = std::chrono::high_resolution_clock::now();
        tree.balance();
        auto end = std::chrono::high_resolution_clock::now();
        
        // tree.verify(); // Check invariants

        std::cout << "Balanced in " 
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() 
                  << "ms. Final size: " << tree.size() << "\n";
    }

    return 0;
}