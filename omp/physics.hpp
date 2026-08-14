/**
 * @file physics.hpp
 * @brief Physics simulation configurations and Refinement Oracles.
 * * @details
 * Oracles are functional predicates (concepts: `RefinementOracle`) that drive the AMR process.
 * - **SphereOracle/CircleOracle**: Geometric criteria checking if a cell intersects a surface.
 */

#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

#include "core.hpp"
#include "tree.hpp"

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
        for (int i = 0; i < dim; ++i)
            c.push_back(static_cast<uint64_t>(center[i] * static_cast<double>(w)));
        return c;
    }
};

class SphereOracle {
    uint64_t cx, cy, cz;
    double radius_dbl, bandwidth_dbl;
    int l_coarse, l_fine;

public:
    SphereOracle(const Config& cfg) : l_coarse(cfg.coarse_level), l_fine(cfg.fine_level) {
        auto c = cfg.int_center(3);
        cx = c[0];
        cy = c[1];
        cz = c[2];
        double w = static_cast<double>(cfg.width());
        radius_dbl = cfg.radius * w;
        bandwidth_dbl = cfg.bandwidth * w;
    }

    bool operator()(const Node& node, int max_lvl) const {
        if (node.level < l_coarse)
            return true;
        if (node.level >= l_fine)
            return false;

        auto [x, y, z] = morton::decode_3d(node.code);
        uint64_t size = 1ULL << (max_lvl - node.level);

        double node_cx = static_cast<double>(x.value) + static_cast<double>(size) * 0.5;
        double node_cy = static_cast<double>(y.value) + static_cast<double>(size) * 0.5;
        double node_cz = static_cast<double>(z.value) + static_cast<double>(size) * 0.5;

        double dx = node_cx - static_cast<double>(cx);
        double dy = node_cy - static_cast<double>(cy);
        double dz = node_cz - static_cast<double>(cz);
        double dist_sq = dx * dx + dy * dy + dz * dz;

        // Updated constant
        double extent = static_cast<double>(size) * amr::constants::SQRT3_OVER_2;
        double threshold = bandwidth_dbl + extent;
        double upper = radius_dbl + threshold;
        double lower = radius_dbl - threshold;
        if (lower < 0)
            lower = 0;

        return (dist_sq < upper * upper && dist_sq > lower * lower);
    }
};

class CircleOracle {
    uint64_t cx, cy;
    double radius_dbl, bandwidth_dbl;
    int l_coarse, l_fine;

public:
    CircleOracle(const Config& cfg) : l_coarse(cfg.coarse_level), l_fine(cfg.fine_level) {
        auto c = cfg.int_center(2);
        cx = c[0];
        cy = c[1];
        double w = static_cast<double>(cfg.width());
        radius_dbl = cfg.radius * w;
        bandwidth_dbl = cfg.bandwidth * w;
    }

    bool operator()(const Node& node, int max_lvl) const {
        if (node.level < l_coarse)
            return true;
        if (node.level >= l_fine)
            return false;

        auto [x, y] = morton::decode_2d(node.code);
        uint64_t size = 1ULL << (max_lvl - node.level);

        double node_cx = static_cast<double>(x.value) + static_cast<double>(size) * 0.5;
        double node_cy = static_cast<double>(y.value) + static_cast<double>(size) * 0.5;

        double dx = node_cx - static_cast<double>(cx);
        double dy = node_cy - static_cast<double>(cy);
        double dist_sq = dx * dx + dy * dy;

        // Updated constant
        double extent = static_cast<double>(size) * amr::constants::SQRT2_OVER_2;
        double threshold = bandwidth_dbl + extent;
        double upper = radius_dbl + threshold;
        double lower = radius_dbl - threshold;
        if (lower < 0)
            lower = 0;

        return (dist_sq < upper * upper && dist_sq > lower * lower);
    }
};

/**
 * @brief Error-driven refinement oracle: Dörfler (bulk) marking from a per-leaf
 *        scalar field (e.g. the DIC residual), with hysteretic coarsening.
 *
 * Consumes the Morton-sorted (codes, levels, error) of the *current* leaf set --
 * exactly the order `mesh_io::write` emits and tree iteration produces, so the DIC
 * solver's per-leaf error array lines up 1:1. Marking is computed once, at
 * construction:
 *   - **refine**: the smallest set of leaves whose summed squared error covers a
 *     fraction `theta_refine` of the total (Dörfler/bulk). A leaf is refined when
 *     its error is at or above the resulting cutoff.
 *   - **coarsen**: the symmetric low-error tail covering `theta_coarsen` of the
 *     total squared error. A complete sibling family is coarsened when *every*
 *     child is at or below the (lower) coarsen cutoff.
 * Keep `theta_refine + theta_coarsen <= 1` so the two bands do not overlap.
 *
 * Refinement and coarsening are *separate* functors -- `refine()` marks leaves,
 * `coarsen()` marks parents with the inverted (false = coarsen) contract the tree
 * expects -- because a single predicate cannot tell a coarsen query on a real
 * parent from a refine query on a finer descendant (both miss the leaf array),
 * which would refine without bound under `while (tree.refine(oracle))`. Use as:
 * @code
 *   ScalarFieldOracle<DIM> oracle(codes, levels, err, 0.3, 0.1, coarse, fine);
 *   while (tree.refine(oracle));          // one Dörfler level of refinement
 *   tree.coarsen(oracle.coarsener());     // one level of low-error coarsening
 * @endcode
 * Each DIC solve produces a fresh error array and a fresh oracle: refinement is
 * one level per marking, matching the solve/mark/refine adaptive loop.
 */
template <int DIM>
class ScalarFieldOracle {
    static constexpr int kChildren = 1 << DIM;

    std::vector<uint64_t> codes_;  // Morton-sorted leaf codes (unique per value)
    std::vector<uint8_t> levels_;  // aligned with codes_
    std::vector<double> err_;      // aligned with codes_
    double refine_cutoff_ = std::numeric_limits<double>::infinity();
    double coarsen_cutoff_ = -std::numeric_limits<double>::infinity();
    int coarse_level_, fine_level_;

    // Index of the leaf with this exact (code, level); -1 if absent. Leaf codes
    // are unique per value, so one lower_bound resolves it.
    std::ptrdiff_t find(uint64_t code, int level) const {
        auto it = std::lower_bound(codes_.begin(), codes_.end(), code);
        if (it == codes_.end() || *it != code)
            return -1;
        std::ptrdiff_t idx = it - codes_.begin();
        return levels_[idx] == level ? idx : -1;
    }

    // Worst (max) error over a parent's kChildren children, or +inf if any child
    // is not a direct leaf -- then the family is not a clean candidate, so it is
    // never coarsened.
    double region_error(uint64_t parent_code, int parent_level, int max_lvl) const {
        const int child_level = parent_level + 1;
        const uint64_t child_span = uint64_t{1} << (DIM * (max_lvl - child_level));
        double worst = 0.0;
        for (int j = 0; j < kChildren; ++j) {
            std::ptrdiff_t idx = find(parent_code + uint64_t(j) * child_span, child_level);
            if (idx < 0)
                return std::numeric_limits<double>::infinity();
            worst = std::max(worst, err_[idx]);
        }
        return worst;
    }

    // The error cutoff whose high (from_top) or low tail covers `theta` of the
    // total squared error. from_top: mark err >= cutoff; else mark err <= cutoff.
    double dorfler_cutoff(double theta, bool from_top) const {
        double total = 0.0;
        for (double e : err_)
            total += e * e;
        if (total == 0.0 || theta <= 0.0)
            return from_top ? std::numeric_limits<double>::infinity()
                            : -std::numeric_limits<double>::infinity();
        std::vector<double> s(err_);
        if (from_top)
            std::sort(s.begin(), s.end(), std::greater<double>());
        else
            std::sort(s.begin(), s.end());
        double acc = 0.0, cutoff = s.front();
        for (double e : s) {
            acc += e * e;
            cutoff = e;
            if (acc >= theta * total)
                break;
        }
        return cutoff;
    }

public:
    ScalarFieldOracle(std::vector<uint64_t> codes,
                      std::vector<uint8_t> levels,
                      std::vector<double> error,
                      double theta_refine,
                      double theta_coarsen,
                      int coarse_level,
                      int fine_level)
        : codes_(std::move(codes)),
          levels_(std::move(levels)),
          err_(std::move(error)),
          coarse_level_(coarse_level),
          fine_level_(fine_level) {
        if (codes_.size() != levels_.size() || codes_.size() != err_.size())
            throw std::invalid_argument("ScalarFieldOracle: codes/levels/error size mismatch");
        if (!std::is_sorted(codes_.begin(), codes_.end()))
            throw std::invalid_argument("ScalarFieldOracle: codes must be Morton-sorted");
        if (theta_refine < 0.0 || theta_refine > 1.0 || theta_coarsen < 0.0 || theta_coarsen > 1.0)
            throw std::invalid_argument("ScalarFieldOracle: theta must be in [0, 1]");
        if (coarse_level_ > fine_level_)
            throw std::invalid_argument(
                "ScalarFieldOracle: coarse_level must not exceed fine_level");
        // A DIC error metric is non-negative; the Dörfler bulk accumulates squared
        // error but marks on the raw value, so a negative (or non-finite) entry
        // would break that correspondence. Reject it rather than mismark.
        for (double e : err_)
            if (!std::isfinite(e) || e < 0.0)
                throw std::invalid_argument(
                    "ScalarFieldOracle: error must be finite and non-negative");
        refine_cutoff_ = dorfler_cutoff(theta_refine, /*from_top=*/true);
        coarsen_cutoff_ = dorfler_cutoff(theta_coarsen, /*from_top=*/false);
    }

    // Refinement oracle (true = refine this leaf). Safe under while(refine()): a
    // node that is not a current high-error leaf is never split (except to reach
    // the base `coarse_level`).
    bool operator()(const Node& n, int max_lvl) const {
        (void)max_lvl;
        if (n.level < coarse_level_)
            return true;  // keep a base resolution everywhere
        if (n.level >= fine_level_)
            return false;  // never exceed the finest level
        const std::ptrdiff_t idx = find(n.code.value, n.level);
        return idx >= 0 && err_[idx] >= refine_cutoff_;
    }

    // Coarsening oracle for tree.coarsen(): called on a parent, false = coarsen.
    // Coarsens a complete family only when its worst child is at or below the
    // (lower) coarsen cutoff, and never below the base `coarse_level`.
    class Coarsener {
        const ScalarFieldOracle* o_;

    public:
        explicit Coarsener(const ScalarFieldOracle* o) : o_(o) {}
        bool operator()(const Node& parent, int max_lvl) const {
            if (parent.level < o_->coarse_level_)
                return true;  // keep the base floor
            return o_->region_error(parent.code.value, parent.level, max_lvl) > o_->coarsen_cutoff_;
        }
    };

    Coarsener coarsener() const { return Coarsener(this); }
};

}  // namespace amr
