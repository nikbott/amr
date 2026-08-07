/**
 * @file viz.hpp
 * @brief MPI-backend adapter over the shared writers in common/viz.hpp.
 *
 * Each rank writes only the leaves it owns, so callers pass a rank-qualified
 * filename and the per-rank files are assembled by the viewer.
 */
#pragma once
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "../common/viz.hpp"
#include "tree.hpp"

namespace amr::viz {

namespace detail {

/// Flatten a DistributedTree's *local* leaves into the arrays the writers take.
template <typename Tree>
struct HostLeaves {
    std::vector<uint64_t> codes;
    std::vector<uint8_t> levels;

    explicit HostLeaves(const Tree& tree) {
        codes.reserve(tree.local_size());
        levels.reserve(tree.local_size());
        for (const auto& node : tree) {
            codes.push_back(node.code.value);
            levels.push_back(static_cast<uint8_t>(node.level));
        }
    }

    LeafView view(const Tree& tree) const {
        constexpr int DIM = static_cast<int>(std::tuple_size_v<typename Tree::Point>);
        return LeafView{codes.data(), levels.data(), codes.size(), tree.max_level, DIM};
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
