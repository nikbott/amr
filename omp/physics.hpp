/**
 * @file physics.hpp
 * @brief Physics simulation configurations and Refinement Oracles.
 * * @details 
 * Oracles are functional predicates (concepts: `RefinementOracle`) that drive the AMR process.
 * - **SphereOracle/CircleOracle**: Geometric criteria checking if a cell intersects a surface.
 */

#pragma once
#include "tree.hpp"
#include "core.hpp"
#include <vector>
#include <cmath>

namespace amr {

struct Config {
    int max_level = 21;
    double radius = 0.25;
    double bandwidth = 0.05;
    std::vector<double> center = {0.5, 0.5, 0.5};
    int coarse_level = 3;
    int fine_level = 9;

    [[nodiscard]] uint64_t width() const { return 1ULL << max_level; }
    
    [[nodiscard]] std::vector<uint64_t> int_center(int dim) const {
        uint64_t w = width();
        std::vector<uint64_t> c;
        for(int i=0; i<dim; ++i) c.push_back(static_cast<uint64_t>(center[i] * static_cast<double>(w)));
        return c;
    }
};

class SphereOracle {
    uint64_t cx, cy, cz;
    double radius_dbl, bandwidth_dbl; 
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
        if (node.level < l_coarse) return true;
        if (node.level >= l_fine) return false;

        auto [x, y, z] = morton::decode_3d(node.code);
        uint64_t size = 1ULL << (max_lvl - node.level);
        
        double node_cx = static_cast<double>(x.value) + static_cast<double>(size) * 0.5;
        double node_cy = static_cast<double>(y.value) + static_cast<double>(size) * 0.5;
        double node_cz = static_cast<double>(z.value) + static_cast<double>(size) * 0.5;
        
        double dx = node_cx - static_cast<double>(cx);
        double dy = node_cy - static_cast<double>(cy);
        double dz = node_cz - static_cast<double>(cz);
        double dist_sq = dx*dx + dy*dy + dz*dz;

        // Updated constant
        double extent = static_cast<double>(size) * amr::constants::SQRT3_OVER_2; 
        double threshold = bandwidth_dbl + extent;
        double upper = radius_dbl + threshold;
        double lower = radius_dbl - threshold;
        if (lower < 0) lower = 0;

        return (dist_sq < upper*upper && dist_sq > lower*lower);
    }
};

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
        
        double node_cx = static_cast<double>(x.value) + static_cast<double>(size) * 0.5;
        double node_cy = static_cast<double>(y.value) + static_cast<double>(size) * 0.5;
        
        double dx = node_cx - static_cast<double>(cx);
        double dy = node_cy - static_cast<double>(cy);
        double dist_sq = dx*dx + dy*dy;

        // Updated constant
        double extent = static_cast<double>(size) * amr::constants::SQRT2_OVER_2;
        double threshold = bandwidth_dbl + extent;
        double upper = radius_dbl + threshold;
        double lower = radius_dbl - threshold;
        if (lower < 0) lower = 0;

        return (dist_sq < upper*upper && dist_sq > lower*lower);
    }
};

} // namespace amr