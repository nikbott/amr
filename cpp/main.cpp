#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstring>
#include <fstream>
#include "config.hpp"
#include "tree.hpp"
#include "physics.hpp"

#ifdef USE_CUDA
#include "cuda_utils.cuh"
#endif

class MeshVisualizer {
public:
    // Exports the Quadtree to an SVG file viewable in any browser
    static void save_svg(const Quadtree& tree, const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return;
        }

        uint64_t limit = tree.domain_width();
        double scale_factor = 1000.0 / limit; // Map domain to 1000x1000 pixels

        // SVG Header
        file << "<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" "
             << "width=\"1000\" height=\"1000\" viewBox=\"0 0 1000 1000\">\n";
        
        // Background
        file << "<rect width=\"1000\" height=\"1000\" fill=\"white\"/>\n";

        // Draw Leaves
        for (const auto& node : tree.leaves) {
            auto [coords, size] = tree.get_geometry(node);
            
            // SVG coordinate system has (0,0) at top-left. 
            // We need to flip Y to match standard cartesian (bottom-left).
            double x = coords[0] * scale_factor;
            double raw_y = coords[1] * scale_factor;
            double h = size * scale_factor;
            double y = 1000.0 - raw_y - h; // Flip Y

            file << "<rect x=\"" << x << "\" y=\"" << y 
                 << "\" width=\"" << h << "\" height=\"" << h 
                 << "\" style=\"fill:none;stroke:red;stroke-width:0.5\" />\n";
        }

        // Add Title/Info text
        file << "<text x=\"10\" y=\"25\" font-family=\"Arial\" font-size=\"20\" fill=\"black\">"
             << "Elements: " << tree.leaves.size() << "</text>\n";

        file << "</svg>";
        file.close();
        std::cout << "[Viz] Saved 2D mesh to " << filename << std::endl;
    }
};

int main(int argc, char** argv) {
    bool use_cuda = false;
    
    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--cuda") == 0 || strcmp(argv[i], "--gpu") == 0) {
            use_cuda = true;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --cuda, --gpu    Use CUDA GPU acceleration" << std::endl;
            std::cout << "  --help, -h       Show this help message" << std::endl;
            return 0;
        }
    }
    
#ifdef USE_CUDA
    if (use_cuda) {
        std::cout << "--- C++ AMR with CUDA Acceleration ---" << std::endl;
        cuda_utils::printGPUInfo();
        std::cout << std::endl;
    } else {
        std::cout << "--- C++ AMR (CPU Mode) ---" << std::endl;
    }
#else
    if (use_cuda) {
        std::cout << "WARNING: CUDA requested but not compiled. Running in CPU mode." << std::endl;
    }
    use_cuda = false;
    std::cout << "--- C++ AMR (CPU Only) ---" << std::endl;
#endif
    
    AMRConfig cfg;
    cfg.max_level = 15; 
    cfg.fine_level = 9;
    cfg.coarse_level = 3;
    cfg.center = {0.5, 0.5};
    cfg.radius = 0.25;

    std::cout << "Config: 2D Quadtree, MaxLvl=" << cfg.max_level << std::endl;
    std::cout << "Mode: " << (use_cuda ? "GPU" : "CPU") << std::endl;

    Quadtree tree(cfg.max_level, use_cuda);
    CircleOracle2D oracle(cfg);

    int max_steps = 16;
    
    std::cout << std::left << std::setw(10) << "Step" 
              << "| " << std::setw(15) << "Elements" 
              << "| " << std::setw(12) << "Time (ms)" << std::endl;
    std::cout << std::string(45, '-') << std::endl;

    auto total_start = std::chrono::high_resolution_clock::now();
    
    for(int step = 0; step < max_steps; ++step) {
        auto step_start = std::chrono::high_resolution_clock::now();
        
        bool changed = tree.refine(oracle);
        
        auto step_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(step_end - step_start);
        
        std::cout << std::left << std::setw(10) << step 
                  << "| " << std::setw(15) << tree.leaves.size() 
                  << "| " << std::setw(12) << duration.count() << std::endl;

        if (!changed) {
            std::cout << "Converged early." << std::endl;
            break;
        }
    }

    std::cout << "\nRunning Balance Constraint..." << std::endl;
    size_t before = tree.leaves.size();
    
    auto balance_start = std::chrono::high_resolution_clock::now();
    tree.balance();
    auto balance_end = std::chrono::high_resolution_clock::now();
    
    auto balance_duration = std::chrono::duration_cast<std::chrono::milliseconds>(balance_end - balance_start);
    size_t after = tree.leaves.size();
    
    std::cout << "Balance complete. Elements: " << before << " -> " << after << std::endl;
    std::cout << "Balance time: " << balance_duration.count() << " ms" << std::endl;
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
    
    std::cout << "\nTotal execution time: " << total_duration.count() << " ms" << std::endl;
    
    // Save SVG
    MeshVisualizer::save_svg(tree, "mesh_2d.svg");

    return 0;
}
