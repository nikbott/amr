/**
 * @file viz.cuh
 * @brief Visualization utilities (SVG and VTK).
 */
#pragma once
#include "tree.cuh"
#include <fstream>
#include <string>
#include <iostream>
#include <vector>

namespace amr::viz {

template <typename Tree>
void write_svg(const Tree& tree, const std::string& filename) {
    // Copy to host
    thrust::host_vector<uint64_t> h_codes = tree.codes;
    thrust::host_vector<uint8_t> h_levels = tree.levels;
    
    std::ofstream f(filename);
    f << "<svg width=\"800\" height=\"800\" viewBox=\"0 0 1000 1000\" xmlns=\"http://www.w3.org/2000/svg\">\n";
    f << "<rect width=\"1000\" height=\"1000\" fill=\"white\"/>\n";
    
    double scale = 1000.0 / (double)(1ULL << tree.max_level);

    for (size_t i = 0; i < h_codes.size(); ++i) {
        Coordinate x, y;
        morton::decode_2d(MortonCode{h_codes[i]}, x, y);
        
        double xx = (double)x.value * scale;
        double yy = 1000.0 - ((double)y.value * scale); // Flip Y
        double s = (double)(1ULL << (tree.max_level - h_levels[i])) * scale;
        
        f << "<rect x=\"" << xx << "\" y=\"" << (yy - s) 
          << "\" width=\"" << s << "\" height=\"" << s 
          << "\" fill=\"none\" stroke=\"red\" stroke-width=\"0.5\"/>\n";
    }
    f << "</svg>\n";
    std::cout << "Wrote " << filename << " (" << h_codes.size() << " elements)\n";
}

template <typename Tree>
void write_vtk(const Tree& tree, const std::string& filename) {
    // Copy to host
    thrust::host_vector<uint64_t> h_codes = tree.codes;
    thrust::host_vector<uint8_t> h_levels = tree.levels;
    size_t n = h_codes.size();

    std::ofstream f(filename);
    f << "# vtk DataFile Version 3.0\nAMR Mesh\nASCII\nDATASET UNSTRUCTURED_GRID\n";
    
    f << "POINTS " << n * 8 << " double\n";
    
    double norm = 1.0 / (double)(1ULL << tree.max_level);
    
    for (size_t i = 0; i < n; ++i) {
        Coordinate cx, cy, cz;
        
        // Use if constexpr to avoid instantiating invalid decode calls
        if constexpr (Tree::dim == 3) {
            morton::decode_3d(MortonCode{h_codes[i]}, cx, cy, cz);
        } else {
            morton::decode_2d(MortonCode{h_codes[i]}, cx, cy);
            cz.value = 0;
        }

        double x = (double)cx.value * norm;
        double y = (double)cy.value * norm;
        double z = (double)cz.value * norm;
        double s = (double)(1ULL << (tree.max_level - h_levels[i])) * norm;

        // 8 Corners
        f << x << " " << y << " " << z << "\n";
        f << x+s << " " << y << " " << z << "\n";
        f << x << " " << y+s << " " << z << "\n";
        f << x+s << " " << y+s << " " << z << "\n";
        f << x << " " << y << " " << z+s << "\n";
        f << x+s << " " << y << " " << z+s << "\n";
        f << x << " " << y+s << " " << z+s << "\n";
        f << x+s << " " << y+s << " " << z+s << "\n";
    }

    f << "\nCELLS " << n << " " << 9*n << "\n";
    for(size_t i=0; i<n; ++i) {
        size_t base = i*8;
        f << "8 " << base << " " << base+1 << " " << base+3 << " " << base+2 << " " 
                 << base+4 << " " << base+5 << " " << base+7 << " " << base+6 << "\n";
    }

    f << "\nCELL_TYPES " << n << "\n";
    for(size_t i=0; i<n; ++i) f << "11\n"; // 11 = Voxel

    std::cout << "Wrote " << filename << " (" << n << " cells)\n";
}

}