/**
 * @file marking.hpp
 * @brief Which elements to refine this cycle, from per-element error indicators.
 *
 * @details Works on plain per-element arrays, so it serves any mesh: the rules
 * the FE-DIC solver applies every cycle, in this order.
 *
 *  1. **flagged**: the estimator's own criterion, `ratio > 1` (xi > 1).
 *  2. **candidates**: flagged elements whose children stay above the size
 *     floor, `length > 2 * min_element_length` (red refinement halves the
 *     length). With no candidate left the floor is exhausted.
 *  3. **marked**: Dörfler bulk marking [Doerfler1996]. Candidates are sorted by
 *     descending squared error (stable: equal errors keep their index order),
 *     and the shortest prefix whose squared errors reach `theta` of the sum
 *     over the *whole* mesh is marked. When no prefix reaches it (late in a
 *     run, when few elements remain flagged), every candidate is marked.
 *  4. **selected**: the marked prefix, cut to `ceil(max_refine_fraction * N)`
 *     elements (a per-cycle safety cap), then to as many as keep the refined
 *     mesh at or below `max_elements` given each element's growth when
 *     refined (children - 1). If not even one fits, the ceiling is reached.
 *
 * The arithmetic follows the MATLAB solver it replaces: the total is summed
 * in index order, the prefix sums are accumulated in sorted order and each is
 * divided by the total before the comparison. MATLAB may sum a long vector in
 * blocks, and computes the length with its own power function, so a prefix
 * within one ulp of `theta`, or a length within one ulp of the floor, could
 * still be decided differently; none of the 235 recorded report cycles is.
 *
 * Parameters behave as the solver's did: theta > 1 marks every candidate, a
 * non-positive floor or ceiling is off, and a fractional ceiling admits the
 * parents whose growth fits in it. Errors must be finite (a NaN error is an
 * estimator bug, not an input to mark around).
 *
 * The octree's ScalarFieldOracle (omp/physics.hpp) keeps its own
 * threshold-based Dörfler rule, with coarsening; this module is the FE-DIC
 * solver's.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace amr::marking {

using Index = std::int32_t;

struct Params {
    double theta = 0.30;                ///< Dörfler fraction of the total squared error
    double max_refine_fraction = 0.30;  ///< per-cycle cap, as a fraction of the elements
    double min_element_length = 0.0;    ///< size floor; <= 0 disables it
    double max_elements = 0.0;          ///< element ceiling; <= 0 disables it
};

enum class Status { selected, no_candidates, floor_exhausted, element_ceiling };

/// The element sets of each stage, in the order the solver records them.
struct Marking {
    Status status = Status::no_candidates;
    std::vector<Index> flagged;     ///< ascending
    std::vector<Index> candidates;  ///< ascending
    std::vector<Index> marked;      ///< by descending error
    std::vector<Index> selected;    ///< a prefix of marked
};

/**
 * @param error   per-element error e_i (finite, >= 0)
 * @param ratio   per-element refinement ratio; > 1 flags the element
 * @param length  per-element characteristic length V^(1/d); may be empty when
 *                the floor is off
 * @param growth  per-element net elements added by refining it (children - 1)
 */
[[nodiscard]] inline Marking mark(std::span<const double> error,
                                  std::span<const double> ratio,
                                  std::span<const double> length,
                                  std::span<const Index> growth,
                                  const Params& p) {
    const std::size_t n = error.size();
    const bool floor_on = p.min_element_length > 0.0;
    if (ratio.size() != n || growth.size() != n || (floor_on && length.size() != n))
        throw std::invalid_argument("marking: error, ratio, length and growth differ in size");
    if (!std::all_of(error.begin(), error.end(), [](double e) { return std::isfinite(e); }))
        throw std::invalid_argument("marking: errors must be finite");
    if (!(p.theta >= 0.0) || !(p.max_refine_fraction > 0.0 && p.max_refine_fraction <= 1.0) ||
        !std::isfinite(p.min_element_length) || !std::isfinite(p.max_elements))
        throw std::invalid_argument("marking: parameter out of range");

    Marking m;
    for (std::size_t i = 0; i < n; ++i)
        if (ratio[i] > 1.0)
            m.flagged.push_back(static_cast<Index>(i));
    if (m.flagged.empty())
        return m;  // no_candidates

    m.candidates = m.flagged;
    if (floor_on) {
        const double child_floor = 2.0 * p.min_element_length;
        std::erase_if(m.candidates, [&](Index e) {
            return !(length[static_cast<std::size_t>(e)] > child_floor);
        });
        if (m.candidates.empty()) {
            m.status = Status::floor_exhausted;
            return m;
        }
    }

    double total = 0.0;
    for (double e : error)
        total += e * e;
    std::vector<Index> order = m.candidates;
    std::stable_sort(order.begin(), order.end(), [&](Index a, Index b) {
        const double ea = error[static_cast<std::size_t>(a)];
        const double eb = error[static_cast<std::size_t>(b)];
        return ea * ea > eb * eb;
    });
    std::size_t n_marked = order.size();
    double cumulative = 0.0;
    for (std::size_t k = 0; k < order.size(); ++k) {
        const double e = error[static_cast<std::size_t>(order[k])];
        cumulative += e * e;
        if (cumulative / total >= p.theta) {
            n_marked = k + 1;
            break;
        }
    }
    m.marked.assign(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(n_marked));

    m.selected = m.marked;
    const auto cap =
        static_cast<std::size_t>(std::ceil(p.max_refine_fraction * static_cast<double>(n)));
    if (m.selected.size() > cap)
        m.selected.resize(cap);

    if (p.max_elements > 0.0) {
        const double budget = p.max_elements - static_cast<double>(n);
        double used = 0.0;
        std::size_t admitted = 0;
        for (Index e : m.selected) {
            used += growth[static_cast<std::size_t>(e)];
            if (used > budget)
                break;
            ++admitted;
        }
        if (admitted == 0) {
            m.status = Status::element_ceiling;
            m.selected.clear();  // nothing was selected
            return m;
        }
        m.selected.resize(admitted);
    }
    m.status = Status::selected;
    return m;
}

}  // namespace amr::marking
