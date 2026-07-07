/**
 * @file main.cpp
 * @brief Main Entry Point.
 */
#include <chrono>
#include <iostream>
#include <string>

#include "core.hpp"
#include "physics.hpp"
#include "tree.hpp"
#include "viz.hpp"

using namespace amr;

int main(int argc, char** argv) {
    amr::mpi::Context mpi(argc, argv);
    if (mpi.rank == 0)
        std::cout << "=== AMR System (SoA + Strong Types) ===\n";

    Config cfg;
    cfg.max_level = 20;
    cfg.coarse_level = 3;
    cfg.fine_level = 10;
    cfg.radius = 0.25;

    {
        if (mpi.rank == 0)
            std::cout << "\n--- 2D Quadtree ---\n";
        DistributedTree<2> tree(cfg.max_level);
        CircleOracle oracle(cfg);
        int steps = 0;
        while (tree.refine(oracle)) {
            tree.repartition();
            size_t n = tree.global_size();
            if (mpi.rank == 0)
                std::cout << "Refine step " << ++steps << ": " << n << " leaves\n";
        }
        if (mpi.rank == 0)
            std::cout << "Balancing...\n";
        tree.repartition();
        auto start = std::chrono::high_resolution_clock::now();
        tree.balance();
        auto end = std::chrono::high_resolution_clock::now();
        tree.verify_global();
        if (mpi.rank == 0) {
            std::cout << "Balanced in "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                      << "ms. Final size: " << tree.global_size() << "\n";
        }
        viz::write_svg(tree, "mesh_2d_rank_" + std::to_string(mpi.rank) + ".svg");
    }

    {
        if (mpi.rank == 0)
            std::cout << "\n--- 3D Octree ---\n";
        DistributedTree<3> tree(cfg.max_level);
        SphereOracle oracle(cfg);
        int steps = 0;
        while (tree.refine(oracle)) {
            tree.repartition();
            size_t n = tree.global_size();
            if (mpi.rank == 0)
                std::cout << "Refine step " << ++steps << ": " << n << " leaves\n";
        }
        if (mpi.rank == 0)
            std::cout << "Balancing...\n";
        tree.repartition();
        auto start = std::chrono::high_resolution_clock::now();
        tree.balance();
        auto end = std::chrono::high_resolution_clock::now();
        if (mpi.rank == 0) {
            std::cout << "Balanced in "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                      << "ms. Final size: " << tree.global_size() << "\n";
        }
        // viz::write_vtk(tree, "mesh_3d_rank_" + std::to_string(mpi.rank) + ".vtk");
    }
    return 0;
}
