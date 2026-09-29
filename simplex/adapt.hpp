/**
 * @file adapt.hpp
 * @brief One refinement cycle of the FE-DIC solver: mark, balance, refine.
 *
 * @details `adapt` takes the estimator's per-element output (error e_i and
 * refinement ratio) and applies the solver's rules (common/marking.hpp): the
 * selected elements are closed under 1-irregular balance and red-refined. A
 * refinement that grows the mesh by less than `min_growth_fraction` is
 * reported as stagnated: the caller correlates that mesh once more and stops.
 *
 * Element length is V^(1/d), the convention of the solver's size floor, its
 * local regularization and the MCOD estimator.
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

/// Area (T3) or volume (T4) of element e.
[[nodiscard]] inline double measure(const Mesh& m, Index e) {
    const auto v = m.element(e);
    const auto p = [&](std::size_t i) { return m.pos[static_cast<std::size_t>(v[i])]; };
    const Point a = p(0), b = p(1), c = p(2);
    const double ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2];
    const double vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2];
    if (m.dim == 2)
        return 0.5 * std::abs(ux * vy - uy * vx);
    const Point d = p(3);
    const double wx = d[0] - a[0], wy = d[1] - a[1], wz = d[2] - a[2];
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
    const std::vector<Index> growth(n, mesh.dim == 2 ? 3 : 7);  // red refinement: 4 or 8 children

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
