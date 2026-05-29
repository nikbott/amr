/**
 * @file main.cu
 * @brief CUDA AMR Main Driver.
 */
#include "tree.cuh"
#include "physics.cuh"
#include "viz.cuh"
#include <iostream>
#include <chrono>

using namespace amr;

int main() {
    std::cout << "=== GPU AMR System (Thrust + CUDA) ===\n";

    Config cfg;
    cfg.max_level = 20;
    cfg.coarse_level = 3;
    cfg.fine_level = 10;
    cfg.radius = 0.25;

    // 2D Simulation
    {
        std::cout << "\n--- 2D Quadtree (GPU) ---\n";
        Quadtree tree(cfg.max_level);
        CircleOracle oracle(cfg);

        int steps = 0;
        float ms = 0;
        
        // Timer
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);

        cudaEventRecord(start);
        while(tree.refine(oracle)) {
            steps++;
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);

        std::cout << "Refined " << steps << " steps in " << ms << "ms. Leaves: " << tree.size() << "\n";

        std::cout << "Balancing...\n";
        cudaEventRecord(start);
        tree.balance();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);

        std::cout << "Balanced in " << ms << "ms. Final size: " << tree.size() << "\n";
        
        tree.verify();
        viz::write_svg(tree, "mesh_2d.svg");
    }

    // 3D Simulation
    {
        std::cout << "\n--- 3D Octree (GPU) ---\n";
        Octree tree(cfg.max_level);
        SphereOracle oracle(cfg);

        int steps = 0;
        while(tree.refine(oracle)) steps++;

        std::cout << "Refined to " << tree.size() << " leaves.\n";

        std::cout << "Balancing...\n";
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        float ms = 0;

        cudaEventRecord(start);
        tree.balance();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);

        std::cout << "Balanced in " << ms << "ms. Final size: " << tree.size() << "\n";
        
        // viz::write_vtk(tree, "mesh_3d.vtk");
    }

    {
        
    }

    return 0;
}