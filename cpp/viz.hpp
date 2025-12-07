#pragma once
#include "tree.hpp"
#include <fstream>
#include <iostream>
#include <string>

class MeshVisualizer {
public:
    static void save_svg(const Quadtree& tree, const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return;
        }

        uint64_t limit = tree.domain_width();
        double scale_factor = 1000.0 / limit;

        file << "<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" "
             << "width=\"1000\" height=\"1000\" viewBox=\"0 0 1000 1000\">\n";
        file << "<rect width=\"1000\" height=\"1000\" fill=\"white\"/>\n";

        for (const auto& node : tree.leaves) {
            auto [coords, size] = tree.get_geometry(node);
            double x = coords[0] * scale_factor;
            double raw_y = coords[1] * scale_factor;
            double h = size * scale_factor;
            double y = 1000.0 - raw_y - h; 

            file << "<rect x=\"" << x << "\" y=\"" << y 
                 << "\" width=\"" << h << "\" height=\"" << h 
                 << "\" style=\"fill:none;stroke:red;stroke-width:0.5\" />\n";
        }

        file << "<text x=\"10\" y=\"25\" font-family=\"Arial\" font-size=\"20\" fill=\"black\">"
             << "Elements: " << tree.leaves.size() << "</text>\n";
        file << "</svg>";
        file.close();
        std::cout << "[Viz] Saved 2D mesh to " << filename << std::endl;
    }

    static void save_obj(const Octree& tree, const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) return;

        file << "# AMR Octree Point Cloud\n";
        uint64_t limit = tree.domain_width();
        double norm = 1.0 / limit; 

        int count = 0;
        for (const auto& node : tree.leaves) {
            if (tree.leaves.size() > 50000 && (count++ % 10 != 0)) continue;

            auto [coords, size] = tree.get_geometry(node);
            double cx = (coords[0] + size * 0.5) * norm;
            double cy = (coords[1] + size * 0.5) * norm;
            double cz = (coords[2] + size * 0.5) * norm;

            file << "v " << cx << " " << cy << " " << cz << "\n";
        }
        file.close();
        std::cout << "[Viz] Saved 3D point cloud to " << filename << std::endl;
    }
};