/**
 * @file adapt.hpp
 * @brief One refinement cycle of the FE-DIC solver: mark, balance, refine.
 *
 * @details `adapt` takes the estimator's per-element output (error e_i and
 * refinement ratio) and applies the solver's rules (common/marking.hpp): the
 * selected elements are closed under 1-irregular balance and red-refined. The
 * element ceiling bounds the refined mesh, balance included: when the closure
 * of the selection would exceed it, the selection is cut to its longest prefix
 * whose closure fits. A refinement that grows the mesh by less than
 * `min_growth_fraction` is reported as stagnated: the caller correlates that
 * mesh once more and stops.
 *
 * Element length is V^(1/d), the convention of the solver's size floor, its
 * local regularization and the MCOD estimator. It is the same for every
 * shape: the floor bounds how many pixels or voxels an element integrates
 * over, which is its measure, so on hybrid meshes it is a volume floor.
 */
#pragma once

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "../common/marking.hpp"
#include "simplex.hpp"

namespace amr::simplex {

enum class AdaptStatus : std::int32_t {
    refined = 0,
    stagnated = 1,
    no_candidates = 2,
    floor_exhausted = 3,
    element_ceiling = 4,
};

struct AdaptParams {
    marking::Params marking;
    double min_growth_fraction = 0.0;  ///< stagnation threshold; 0 disables it
};

struct Adaptation {
    AdaptStatus status = AdaptStatus::no_candidates;
    marking::Marking marking;
    std::vector<Index> balanced;      ///< ascending; empty unless refined or stagnated
    Refinement refinement;            ///< the refined mesh; empty unless refined or stagnated
    std::array<double, 3> seconds{};  ///< wall time of marking, balance and refinement
};

/// Area or volume of element e. Tetrahedra keep the cofactor expansion the
/// solver's size floor was recorded with, so the floor decides the same way
/// bit for bit; every other shape uses signed_measure.
[[nodiscard]] inline double measure(const Mesh& m, Index e) {
    const auto v = m.element(e);
    std::array<Point, 8> x;
    for (std::size_t i = 0; i < v.size(); ++i)
        x[i] = m.pos[static_cast<std::size_t>(v[i])];
    const Shape shape = m.type[static_cast<std::size_t>(e)];
    if (shape != Shape::T4)
        return std::abs(signed_measure(shape, std::span(x.data(), v.size())));
    const double ux = x[1][0] - x[0][0], uy = x[1][1] - x[0][1], uz = x[1][2] - x[0][2];
    const double vx = x[2][0] - x[0][0], vy = x[2][1] - x[0][1], vz = x[2][2] - x[0][2];
    const double wx = x[3][0] - x[0][0], wy = x[3][1] - x[0][1], wz = x[3][2] - x[0][2];
    return std::abs(ux * (vy * wz - vz * wy) - uy * (vx * wz - vz * wx) +
                    uz * (vx * wy - vy * wx)) /
           6.0;
}

/// Characteristic length V^(1/d) of every element.
[[nodiscard]] inline std::vector<double> element_lengths(const Mesh& m) {
    std::vector<double> length(static_cast<std::size_t>(m.num_elements()));
    const double exponent = 1.0 / m.dim;
    for (Index e = 0; e < m.num_elements(); ++e)
        length[static_cast<std::size_t>(e)] = std::pow(measure(m, e), exponent);
    return length;
}

[[nodiscard]] inline Adaptation adapt(const Mesh& mesh,
                                      std::span<const double> error,
                                      std::span<const double> ratio,
                                      const AdaptParams& p) {
    detail::validate(mesh);
    const auto n = static_cast<std::size_t>(mesh.num_elements());
    if (error.size() != n || ratio.size() != n)
        throw std::invalid_argument("adapt: error and ratio need one value per element");
    if (!(p.min_growth_fraction >= 0.0 && p.min_growth_fraction < 1.0))
        throw std::invalid_argument("adapt: min_growth_fraction must be in [0, 1)");

    using Clock = std::chrono::steady_clock;
    const auto since = [](Clock::time_point t0) {
        return std::chrono::duration<double>(Clock::now() - t0).count();
    };
    auto t0 = Clock::now();
    // Lengths are needed only for the floor.
    const auto length =
        p.marking.min_element_length > 0.0 ? element_lengths(mesh) : std::vector<double>{};
    std::vector<Index> growth(n);  // elements a refinement adds: children - 1
    for (std::size_t e = 0; e < n; ++e)
        growth[e] = static_cast<Index>(element_type(mesh.type[e]).red.children()) - 1;

    Adaptation a;
    a.marking = marking::mark(error, ratio, length, growth, p.marking);
    a.seconds[0] = since(t0);
    switch (a.marking.status) {
        case marking::Status::no_candidates:
            a.status = AdaptStatus::no_candidates;
            return a;
        case marking::Status::floor_exhausted:
            a.status = AdaptStatus::floor_exhausted;
            return a;
        case marking::Status::element_ceiling:
            a.status = AdaptStatus::element_ceiling;
            return a;
        case marking::Status::selected:
            break;
    }
    t0 = Clock::now();
    a.balanced = detail::closure(mesh, a.marking.selected);
    // The marking budgeted the selected elements only; balance can add more.
    // A prefix's closure grows with the prefix, so the longest prefix that
    // fits is found by bisection.
    const auto added = [&growth](std::span<const Index> list) {
        double sum = 0.0;
        for (Index e : list)
            sum += growth[static_cast<std::size_t>(e)];
        return sum;
    };
    const double budget = p.marking.max_elements - static_cast<double>(n);
    if (p.marking.max_elements > 0.0 && added(a.balanced) > budget) {
        auto& selected = a.marking.selected;
        std::size_t fits = 0, exceeds = selected.size();
        std::vector<Index> closed;
        while (exceeds - fits > 1) {
            const std::size_t k = (fits + exceeds) / 2;
            auto c = detail::closure(mesh, std::span(selected.data(), k));
            if (added(c) <= budget) {
                fits = k;
                closed = std::move(c);
            } else {
                exceeds = k;
            }
        }
        selected.resize(fits);
        a.balanced = std::move(closed);
        if (fits == 0) {
            a.marking.status = marking::Status::element_ceiling;
            a.status = AdaptStatus::element_ceiling;
            a.seconds[1] = since(t0);
            return a;
        }
    }
    a.seconds[1] = since(t0);
    t0 = Clock::now();
    a.refinement = detail::refine_valid(mesh, a.balanced);
    a.seconds[2] = since(t0);
    const double grown =
        static_cast<double>(a.refinement.mesh.num_elements() - mesh.num_elements()) /
        static_cast<double>(std::max<std::size_t>(n, 1));
    a.status = p.min_growth_fraction > 0.0 && grown < p.min_growth_fraction ? AdaptStatus::stagnated
                                                                            : AdaptStatus::refined;
    return a;
}

}  // namespace amr::simplex
