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

using namespace std;
using namespace chrono;

class MeshVisualizer {
public:
    // Exports the Quadtree to an SVG file viewable in any browser
    static void save_svg(const Quadtree& tree, const string& filename) {
        ofstream file(filename);
        if (!file.is_open()) {
            cerr << "Error: Could not open file " << filename << endl;
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
        cout << "[Viz] Saved 2D mesh to " << filename << endl;
    }
};

int main(int argc, char** argv) {
    bool use_cuda = false;
    
    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--cuda") == 0 || strcmp(argv[i], "--gpu") == 0) {
            use_cuda = true;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            cout << "Usage: " << argv[0] << " [options]" << endl;
            cout << "Options:" << endl;
            cout << "  --cuda, --gpu    Use CUDA GPU acceleration" << endl;
            cout << "  --help, -h       Show this help message" << endl;
            return 0;
        }
    }
    
#ifdef USE_CUDA
    if (use_cuda) {
        cout << "--- C++ AMR with CUDA Acceleration ---" << endl;
        cuda_utils::printGPUInfo();
        cout << endl;
    } else {
        cout << "--- C++ AMR (CPU Mode) ---" << endl;
    }
#else
    if (use_cuda) {
        cout << "WARNING: CUDA requested but not compiled. Running in CPU mode." << endl;
    }
    use_cuda = false;
    cout << "--- C++ AMR (CPU Only) ---" << endl;
#endif
    
    AMRConfig cfg;
    cfg.max_level = 15; 
    cfg.fine_level = 9;
    cfg.coarse_level = 3;
    cfg.center = {0.5, 0.5};
    cfg.radius = 0.25;

    // Parse configuration arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--max_level") == 0 && i + 1 < argc) {
            cfg.max_level = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--fine_level") == 0 && i + 1 < argc) {
            cfg.fine_level = atoi(argv[++i]);
        }
    }

    cout << "Config: 2D Quadtree, MaxLvl=" << cfg.max_level << endl;
    cout << "Mode: " << (use_cuda ? "GPU" : "CPU") << endl;

    Quadtree tree(cfg.max_level, use_cuda);
    CircleOracle2D oracle(cfg);

    int max_steps = 16;
    
    cout << left << setw(10) << "Step" 
         << "| " << setw(15) << "Elements" 
         << "| " << setw(12) << "Time (ms)" << endl;
    cout << string(45, '-') << endl;

    auto total_start = high_resolution_clock::now();
    
 
    for(int step = 0; step < max_steps; ++step) {
        auto step_start = high_resolution_clock::now();
        
        bool changed = tree.refine(oracle);
        
        auto step_end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(step_end - step_start);
        
#ifdef USE_CUDA
        size_t current_size = use_gpu ? tree.getGPUSize() : tree.leaves.size();
#else
        size_t current_size = tree.leaves.size();
#endif
        
        cout << left << setw(10) << step 
             << "| " << setw(15) << current_size 
             << "| " << setw(12) << duration.count() << endl;

        if (!changed) {
            cout << "Converged early." << endl;
            break;
        }
    }

    cout << "\nRunning Balance Constraint..." << endl;
    
#ifdef USE_CUDA
    size_t before = use_gpu ? tree.getGPUSize() : tree.leaves.size();
#else
    size_t before = tree.leaves.size();
#endif

    auto balance_start = high_resolution_clock::now();
    tree.balance();
    auto balance_end = high_resolution_clock::now();
    
    auto balance_duration = duration_cast<milliseconds>(balance_end - balance_start);

#ifdef USE_CUDA
    size_t after = use_gpu ? tree.getGPUSize() : tree.leaves.size();
#else
    size_t after = tree.leaves.size();
#endif
    
    cout << "Balance complete. Elements: " << before << " -> " << after << endl;
    cout << "Balance time: " << balance_duration.count() << " ms" << endl;
    
    auto total_end = high_resolution_clock::now();
    auto total_duration = duration_cast<milliseconds>(total_end - total_start);
    
    cout << "\nTotal execution time: " << total_duration.count() << " ms" << endl;
    
#ifdef USE_CUDA
    // Sincroniza GPU -> CPU apenas uma vez antes de gerar o SVG
    if (use_gpu) {
        cout << "Syncing GPU data for SVG export..." << endl;
        tree.syncFromGPU();
    }
#endif
    
    // Save SVG
    MeshVisualizer::save_svg(tree, "mesh_2d.svg");

    return 0;
}
