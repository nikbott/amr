#pragma once
#include "tree.hpp"
#include "morton.hpp"
#include <vector>
#include <cmath>

// Decoupled Oracles: Independent of AMRConfig struct
class CircleOracle2D {
    uint64_t cx, cy;
    uint64_t radius;
    uint64_t bandwidth;
    int coarse_level;
    int fine_level;
    
public:
    CircleOracle2D(uint64_t center_x, uint64_t center_y, uint64_t r, uint64_t band, int c_lvl, int f_lvl) 
        : cx(center_x), cy(center_y), radius(r), bandwidth(band), coarse_level(c_lvl), fine_level(f_lvl) {}

    bool operator()(const Node& node, int max_level_grid) const {
        auto [x, y] = Morton2D::decode(node.code);
        uint64_t size = 1ULL << (max_level_grid - node.level);

        double node_cx = x + size * 0.5;
        double node_cy = y + size * 0.5;

        double dx = node_cx - cx;
        double dy = node_cy - cy;
        double dist_sq = dx*dx + dy*dy;

        double extent = size * 0.70710678; 
        double threshold = bandwidth + extent;

        double r_dbl = static_cast<double>(radius);
        double upper = r_dbl + threshold;
        double lower = r_dbl - threshold;
        
        double upper_sq = upper * upper;
        double lower_sq = (lower < 0) ? 0.0 : lower * lower;

        bool is_refining = (dist_sq < upper_sq) && (dist_sq > lower_sq);

        return (node.level < coarse_level) || 
               (is_refining && node.level < fine_level);
    }
};

class SphereOracle3D {
    uint64_t cx, cy, cz;
    uint64_t radius;
    uint64_t bandwidth;
    int coarse_level;
    int fine_level;

public:
    SphereOracle3D(uint64_t center_x, uint64_t center_y, uint64_t center_z, 
                   uint64_t r, uint64_t band, int c_lvl, int f_lvl) 
        : cx(center_x), cy(center_y), cz(center_z), 
          radius(r), bandwidth(band), coarse_level(c_lvl), fine_level(f_lvl) {}

    bool operator()(const Node& node, int max_level_grid) const {
        auto [x, y, z] = Morton3D::decode(node.code);
        uint64_t size = 1ULL << (max_level_grid - node.level);

        double node_cx = x + size * 0.5;
        double node_cy = y + size * 0.5;
        double node_cz = z + size * 0.5;

        double dx = node_cx - cx;
        double dy = node_cy - cy;
        double dz = node_cz - cz;
        double dist_sq = dx*dx + dy*dy + dz*dz;

        double extent = size * 0.8660254;
        double threshold = bandwidth + extent;

        double r_dbl = static_cast<double>(radius);
        double upper = r_dbl + threshold;
        double lower = r_dbl - threshold;

        double upper_sq = upper * upper;
        double lower_sq = (lower < 0) ? 0.0 : lower * lower;

        bool is_refining = (dist_sq < upper_sq) && (dist_sq > lower_sq);

        return (node.level < coarse_level) || 
               (is_refining && node.level < fine_level);
    }
};