/**
 * @file physics.cuh
 * @brief Refinement Oracles for CUDA (Fixed: Bandwidth & Coarse Constraints).
 */
#pragma once
#include "core.cuh"

namespace amr {

struct Config {
    int max_level = 20;
    int coarse_level = 3;
    int fine_level = 9;
    double radius = 0.25;
    double bandwidth = 0.05; // Added missing field
    double center_x = 0.5, center_y = 0.5, center_z = 0.5;
    
    HOST_DEVICE uint64_t width() const { return 1ULL << max_level; }
};

// --- Oracles (Device Functors) ---

struct CircleOracle {
    uint64_t cx, cy;
    double r_dbl, bw_dbl;
    int l_coarse, l_fine;
    int max_lvl;

    CircleOracle(Config cfg) 
        : l_coarse(cfg.coarse_level), l_fine(cfg.fine_level), max_lvl(cfg.max_level) 
    {
        double w = (double)cfg.width();
        cx = (uint64_t)(cfg.center_x * w);
        cy = (uint64_t)(cfg.center_y * w);
        r_dbl = cfg.radius * w;
        bw_dbl = cfg.bandwidth * w;
    }

    HOST_DEVICE bool operator()(MortonCode code, int level) const {
        // 1. Force refinement if too coarse
        if (level < l_coarse) return true;
        // 2. Stop refinement if fine enough
        if (level >= l_fine) return false;
        
        Coordinate x, y;
        morton::decode_2d(code, x, y);
        uint64_t size = 1ULL << (max_lvl - level);
        
        // Calculate Distance to Center
        double cell_cx = (double)x.value + (double)size * 0.5;
        double cell_cy = (double)y.value + (double)size * 0.5;
        
        double dx = cell_cx - (double)cx;
        double dy = cell_cy - (double)cy;
        double dist_sq = dx*dx + dy*dy;

        // Bandwidth Check (Boundary Refinement)
        // extent = size * sqrt(2)/2
        double extent = (double)size * amr::constants::SQRT2_OVER_2;
        double threshold = bw_dbl + extent;
        
        double upper = r_dbl + threshold;
        double lower = r_dbl - threshold;
        if (lower < 0.0) lower = 0.0;

        return (dist_sq < upper*upper && dist_sq > lower*lower);
    }
};

struct SphereOracle {
    uint64_t cx, cy, cz;
    double r_dbl, bw_dbl;
    int l_coarse, l_fine;
    int max_lvl;

    SphereOracle(Config cfg) 
        : l_coarse(cfg.coarse_level), l_fine(cfg.fine_level), max_lvl(cfg.max_level) 
    {
        double w = (double)cfg.width();
        cx = (uint64_t)(cfg.center_x * w);
        cy = (uint64_t)(cfg.center_y * w);
        cz = (uint64_t)(cfg.center_z * w);
        r_dbl = cfg.radius * w;
        bw_dbl = cfg.bandwidth * w;
    }

    HOST_DEVICE bool operator()(MortonCode code, int level) const {
        if (level < l_coarse) return true;
        if (level >= l_fine) return false;
        
        Coordinate x, y, z;
        morton::decode_3d(code, x, y, z);
        uint64_t size = 1ULL << (max_lvl - level);
        
        double cell_cx = (double)x.value + (double)size * 0.5;
        double cell_cy = (double)y.value + (double)size * 0.5;
        double cell_cz = (double)z.value + (double)size * 0.5;
        
        double dx = cell_cx - (double)cx;
        double dy = cell_cy - (double)cy;
        double dz = cell_cz - (double)cz;
        double dist_sq = dx*dx + dy*dy + dz*dz;
        
        // Bandwidth Check
        double extent = (double)size * amr::constants::SQRT3_OVER_2;
        double threshold = bw_dbl + extent;
        
        double upper = r_dbl + threshold;
        double lower = r_dbl - threshold;
        if (lower < 0.0) lower = 0.0;
        
        return (dist_sq < upper*upper && dist_sq > lower*lower);
    }
};

}