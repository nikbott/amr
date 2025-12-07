#pragma once
#include "tree.hpp"
#include <vector>
#include <cmath>

namespace amr {

/**
 * @brief Configuration for the AMR simulation.
 */
struct Config {
    int max_level = 21;
    double radius = 0.25;
    double bandwidth = 0.05;
    std::vector<double> center = {0.5, 0.5, 0.5};
    int coarse_level = 3;
    int fine_level = 9;

    [[nodiscard]] uint64_t width() const { return 1ULL << max_level; }
    
    // Helper to get integer center coordinates
    [[nodiscard]] std::vector<uint64_t> int_center(int dim) const {
        uint64_t w = width();
        std::vector<uint64_t> c;
        for(int i=0; i<dim; ++i) c.push_back(static_cast<uint64_t>(center[i] * w));
        return c;
    }
};

/**
 * @brief 3D Sphere Refinement Oracle.
 * * Determines if a node intersects a spherical shell defined by radius and bandwidth.
 */
class SphereOracle {
    uint64_t cx, cy, cz;
    double radius_dbl, bandwidth_dbl; // Keep as double for precise extent calculations
    int l_coarse, l_fine;

public:
    SphereOracle(const Config& cfg) 
        : l_coarse(cfg.coarse_level), l_fine(cfg.fine_level) {
        
        auto c = cfg.int_center(3);
        cx = c[0]; cy = c[1]; cz = c[2];
        
        double w = static_cast<double>(cfg.width());
        radius_dbl = cfg.radius * w;
        bandwidth_dbl = cfg.bandwidth * w;
    }

    bool operator()(const Node& node, int max_lvl) const {
        // Enforce strict level bounds
        if (node.level < l_coarse) return true;
        if (node.level >= l_fine) return false;

        auto [x, y, z] = morton::decode_3d(node.code);
        uint64_t size = 1ULL << (max_lvl - node.level);
        
        // Calculate node center
        double node_cx = x + size * 0.5;
        double node_cy = y + size * 0.5;
        double node_cz = z + size * 0.5;
        
        double dx = node_cx - cx;
        double dy = node_cy - cy;
        double dz = node_cz - cz;
        double dist_sq = dx*dx + dy*dy + dz*dz;

        // For a cube of side 'size', the semi-diagonal is size * sqrt(3) / 2.
        // sqrt(3)/2 approx 0.86602540378
        double extent = size * 0.86602540378; 
        
        // We refine if the node's extent overlaps the target band.
        double threshold = bandwidth_dbl + extent;

        double upper = radius_dbl + threshold;
        double lower = radius_dbl - threshold;
        if (lower < 0) lower = 0;

        // Check intersection: dist < upper && dist > lower
        return (dist_sq < upper*upper && dist_sq > lower*lower);
    }
};

/**
 * @brief 2D Circle Refinement Oracle.
 * * Determines if a node intersects a circular ring.
 * Uses bounding circle logic (semi-diagonal of square).
 */
class CircleOracle {
    uint64_t cx, cy;
    double radius_dbl, bandwidth_dbl;
    int l_coarse, l_fine;

public:
    CircleOracle(const Config& cfg) 
        : l_coarse(cfg.coarse_level), l_fine(cfg.fine_level) {
        
        auto c = cfg.int_center(2);
        cx = c[0]; cy = c[1];
        
        double w = static_cast<double>(cfg.width());
        radius_dbl = cfg.radius * w;
        bandwidth_dbl = cfg.bandwidth * w;
    }

    bool operator()(const Node& node, int max_lvl) const {
        if (node.level < l_coarse) return true;
        if (node.level >= l_fine) return false;

        auto [x, y] = morton::decode_2d(node.code);
        uint64_t size = 1ULL << (max_lvl - node.level);
        
        double node_cx = x + size * 0.5;
        double node_cy = y + size * 0.5;
        
        double dx = node_cx - cx;
        double dy = node_cy - cy;
        double dist_sq = dx*dx + dy*dy;

        // For a square of side 'size', the semi-diagonal is size * sqrt(2) / 2.
        // sqrt(2)/2 approx 0.70710678118
        double extent = size * 0.70710678118;
        double threshold = bandwidth_dbl + extent;

        double upper = radius_dbl + threshold;
        double lower = radius_dbl - threshold;
        if (lower < 0) lower = 0;

        return (dist_sq < upper*upper && dist_sq > lower*lower);
    }
};

} // namespace amr