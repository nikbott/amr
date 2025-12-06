#pragma once
#include "config.hpp"
#include "morton.hpp"
#include "tree.hpp"

class CircleOracle2D {
    AMRConfig config;
    uint64_t cx, cy;
    uint64_t radius_sq;
    uint64_t bandwidth;
    
public:
    CircleOracle2D(const AMRConfig& cfg) : config(cfg) {
        auto centers = cfg.get_int_center(2);
        cx = centers[0]; cy = centers[1];
        bandwidth = cfg.get_int_bandwidth();
    }

    bool operator()(const Node& node, int max_level_grid) {
        // 1. Decode
        auto [x, y] = Morton2D::decode(node.code);
        uint64_t size = 1ULL << (max_level_grid - node.level);

        // 2. Node Center (using double for distance calc to be safe with large squares)
        double node_cx = x + size * 0.5;
        double node_cy = y + size * 0.5;

        // 3. Sq Distance
        double dx = node_cx - cx;
        double dy = node_cy - cy;
        double dist_sq = dx*dx + dy*dy;

        // 4. Bounds
        double extent = size * 0.70710678; // size * sqrt(2)/2
        double threshold = bandwidth + extent;

        double r = static_cast<double>(config.get_int_radius());
        double upper = r + threshold;
        double lower = r - threshold;
        
        double upper_sq = upper * upper;
        double lower_sq = (lower < 0) ? 0.0 : lower * lower;

        // 5. Check
        bool is_refining = (dist_sq < upper_sq) && (dist_sq > lower_sq);

        return (node.level < config.coarse_level) || 
               (is_refining && node.level < config.fine_level);
    }
};

class SphereOracle3D {
    AMRConfig config;
    uint64_t cx, cy, cz;
    uint64_t bandwidth;

public:
    SphereOracle3D(const AMRConfig& cfg) : config(cfg) {
        auto centers = cfg.get_int_center(3);
        cx = centers[0]; cy = centers[1]; cz = centers[2];
        bandwidth = cfg.get_int_bandwidth();
    }

    bool operator()(const Node& node, int max_level_grid) {
        auto [x, y, z] = Morton3D::decode(node.code);
        uint64_t size = 1ULL << (max_level_grid - node.level);

        double node_cx = x + size * 0.5;
        double node_cy = y + size * 0.5;
        double node_cz = z + size * 0.5;

        double dx = node_cx - cx;
        double dy = node_cy - cy;
        double dz = node_cz - cz;
        double dist_sq = dx*dx + dy*dy + dz*dz;

        // Extent = size * sqrt(3) / 2
        double extent = size * 0.8660254;
        double threshold = bandwidth + extent;

        double r = static_cast<double>(config.get_int_radius());
        double upper = r + threshold;
        double lower = r - threshold;

        double upper_sq = upper * upper;
        double lower_sq = (lower < 0) ? 0.0 : lower * lower;

        bool is_refining = (dist_sq < upper_sq) && (dist_sq > lower_sq);

        return (node.level < config.coarse_level) || 
               (is_refining && node.level < config.fine_level);
    }
};
