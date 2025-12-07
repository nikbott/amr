#pragma once
#include "tree.hpp"
#include <fstream>
#include <string>
#include <iostream>

namespace amr::viz {

template <typename Tree>
void write_svg(const Tree& tree, const std::string& filename) {
    std::ofstream f(filename);
    f << "<svg width=\"800\" height=\"800\" viewBox=\"0 0 1000 1000\" xmlns=\"http://www.w3.org/2000/svg\">\n";
    f << "<rect width=\"1000\" height=\"1000\" fill=\"white\"/>\n";
    
    double scale = 1000.0 / tree.domain_width();
    
    for (const auto& node : tree.leaves) {
        auto coords = tree.decode(node.code);
        double x = coords[0] * scale;
        double y = 1000.0 - (coords[1] * scale); // Flip Y
        double s = (1ULL << (tree.max_level - node.level)) * scale;
        
        f << "<rect x=\"" << x << "\" y=\"" << (y - s) 
          << "\" width=\"" << s << "\" height=\"" << s 
          << "\" fill=\"none\" stroke=\"red\" stroke-width=\"0.5\"/>\n";
    }
    f << "</svg>\n";
    std::cout << "Wrote " << filename << " (" << tree.leaves.size() << " elements)\n";
}

template <typename Tree>
void write_vtk(const Tree& tree, const std::string& filename) {
    std::ofstream f(filename);
    f << "# vtk DataFile Version 3.0\nAMR Mesh\nASCII\nDATASET UNSTRUCTURED_GRID\n";
    
    size_t n = tree.leaves.size();
    // For 3D Octree, we output points (centers)
    f << "POINTS " << n << " double\n";
    
    double norm = 1.0 / tree.domain_width();
    
    for (const auto& node : tree.leaves) {
        auto coords = tree.decode(node.code);
        uint64_t half = (1ULL << (tree.max_level - node.level)) / 2;
        double x = (coords[0] + half) * norm;
        double y = (coords[1] + half) * norm;
        double z = (coords[2] + half) * norm;
        f << x << " " << y << " " << z << "\n";
    }
    
    f << "\nCELLS " << n << " " << 2*n << "\n";
    for(size_t i=0; i<n; ++i) f << "1 " << i << "\n";
    
    f << "\nCELL_TYPES " << n << "\n";
    for(size_t i=0; i<n; ++i) f << "1\n"; // Vertex
    
    std::cout << "Wrote " << filename << " (" << n << " points)\n";
}

}