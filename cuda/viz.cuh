/**
 * @file viz.cuh
 * @brief CUDA-backend adapter over the shared writers in common/viz.hpp.
 */
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include <thrust/copy.h>

#include "../common/viz.hpp"
#include "tree.cuh"

namespace amr::viz {

namespace detail {

/// Copy a device-resident leaf set to the host arrays the writers take.
template <typename Tree>
struct HostLeaves {
    std::vector<uint64_t> codes;
    std::vector<uint8_t> levels;

    explicit HostLeaves(const Tree& tree) : codes(tree.codes.size()), levels(tree.levels.size()) {
        // One bulk cudaMemcpy per array. Constructing the std::vectors straight
        // from the device iterators would instead dereference element-by-element
        // on the host, issuing O(N) single-element device->host transfers.
        thrust::copy(tree.codes.begin(), tree.codes.end(), codes.begin());
        thrust::copy(tree.levels.begin(), tree.levels.end(), levels.begin());
    }

    LeafView view(const Tree& tree) const {
        return LeafView{codes.data(), levels.data(), codes.size(), tree.max_level, Tree::dim};
    }
};

}  // namespace detail

template <typename Tree>
void write_svg(const Tree& tree, const std::string& filename) {
    const detail::HostLeaves<Tree> leaves(tree);
    write_svg(filename, leaves.view(tree));
}

template <typename Tree>
void write_vtk(const Tree& tree, const std::string& filename) {
    const detail::HostLeaves<Tree> leaves(tree);
    write_vtk(filename, leaves.view(tree));
}

}  // namespace amr::viz
