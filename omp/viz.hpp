/**
 * @file viz.hpp
 * @brief OpenMP-backend adapter over the shared writers in common/viz.hpp.
 */
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "../common/viz.hpp"
#include "tree.hpp"

namespace amr::viz {

namespace detail {

/// Flatten a LinearTree's leaves into the backend-agnostic arrays the writers take.
template <typename Tree>
struct HostLeaves {
    std::vector<uint64_t> codes;
    std::vector<uint8_t> levels;

    explicit HostLeaves(const Tree& tree) {
        codes.reserve(tree.size());
        levels.reserve(tree.size());
        for (const auto& node : tree) {
            codes.push_back(node.code.value);
            levels.push_back(static_cast<uint8_t>(node.level));
        }
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
