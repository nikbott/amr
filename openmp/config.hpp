#pragma once
#include <vector>
#include <cstdint>

struct AMRConfig {
    // Grid Resolution
    int max_level = 21;
    
    // Physics Parameters
    std::vector<double> center = {0.5, 0.5, 0.5};
    double radius = 0.25;
    double bandwidth = 0.05;
    
    // Refinement Limits
    int coarse_level = 4;
    int fine_level = 10;

    uint64_t domain_width() const {
        return 1ULL << max_level;
    }

    // Helper to get integer center based on dimension
    std::vector<uint64_t> get_int_center(int ndim) const {
        uint64_t w = domain_width();
        std::vector<uint64_t> int_c;
        int_c.reserve(ndim);
        for(int i=0; i<ndim; ++i) {
            int_c.push_back(static_cast<uint64_t>(center[i] * w));
        }
        return int_c;
    }

    double get_int_radius() const {
        return radius * domain_width();
    }

    double get_int_bandwidth() const {
        return bandwidth * domain_width();
    }
};
