/**
 * @file test_marking.cpp
 * @brief The solver's marking rules (common/marking.hpp) against oracles built
 * from their definitions.
 *
 * @details Dörfler's criterion [Doerfler1996] asks for a smallest set of
 * elements whose squared errors reach theta of the total; the oracle finds its
 * size by brute force over every subset of the candidates, independently of
 * the greedy sorted prefix the module computes.
 */

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "../common/marking.hpp"

using namespace amr::marking;

namespace {

struct Case {
    std::vector<double> error, ratio, length;
    std::vector<Index> growth;
};

Case uniform_case(std::size_t n, double length = 1.0, Index growth = 3) {
    return {std::vector<double>(n, 1.0),
            std::vector<double>(n, 0.0),
            std::vector<double>(n, length),
            std::vector<Index>(n, growth)};
}

Marking run(const Case& c, const Params& p) {
    return mark(c.error, c.ratio, c.length, c.growth, p);
}

/// Smallest number of candidates whose squared errors reach theta of the total
/// over all elements; SIZE_MAX if no subset does.
std::size_t brute_force_dorfler(const std::vector<double>& error,
                                const std::vector<Index>& candidates,
                                double theta) {
    double total = 0;
    for (double e : error)
        total += e * e;
    std::size_t best = std::numeric_limits<std::size_t>::max();
    const std::size_t n = candidates.size();
    for (std::uint32_t mask = 1; mask < (1u << n); ++mask) {
        double sum = 0;
        for (std::size_t i = 0; i < n; ++i)
            if (mask & (1u << i)) {
                const double e = error[static_cast<std::size_t>(candidates[i])];
                sum += e * e;
            }
        if (sum >= theta * total * (1 - 1e-12))
            best = std::min<std::size_t>(best, static_cast<std::size_t>(std::popcount(mask)));
    }
    return best;
}

}  // namespace

TEST_CASE("Dörfler marks a smallest set of candidates reaching theta", "[marking]") {
    std::mt19937 rng(101);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    int reached = 0;
    for (int trial = 0; trial < 400; ++trial) {
        const auto n = static_cast<std::size_t>(2 + trial % 11);
        Case c = uniform_case(n);
        for (std::size_t i = 0; i < n; ++i) {
            c.error[i] = std::pow(u(rng), 3.0);
            c.ratio[i] = u(rng) < 0.6 ? 2.0 : 0.5;
        }
        Params p;
        p.theta = u(rng);
        p.max_refine_fraction = 1.0;
        const auto m = run(c, p);
        if (m.flagged.empty())
            continue;
        const auto best = brute_force_dorfler(c.error, m.candidates, p.theta);
        if (best == std::numeric_limits<std::size_t>::max()) {
            CHECK(m.marked.size() == m.candidates.size());  // the fallback
        } else {
            CHECK(m.marked.size() == best);
            ++reached;
        }
        // Marked elements are candidates, by non-increasing error.
        for (std::size_t k = 0; k < m.marked.size(); ++k) {
            CHECK(std::binary_search(m.candidates.begin(), m.candidates.end(), m.marked[k]));
            if (k > 0)
                CHECK(c.error[static_cast<std::size_t>(m.marked[k - 1])] >=
                      c.error[static_cast<std::size_t>(m.marked[k])]);
        }
    }
    CHECK(reached > 100);
}

TEST_CASE("Only ratio > 1 flags, and NaN never does", "[marking]") {
    Case c = uniform_case(5);
    c.ratio = {2.0, 1.0, std::nan(""), 1.0000001, 0.0};
    const auto m = run(c, {});
    CHECK(m.flagged == std::vector<Index>{0, 3});
    CHECK(m.status == Status::selected);
}

TEST_CASE("Nothing flagged means no candidates", "[marking]") {
    const auto m = run(uniform_case(4), {});
    CHECK(m.status == Status::no_candidates);
    CHECK(m.selected.empty());
}

TEST_CASE("The floor keeps only elements whose children stay above it", "[marking]") {
    Case c = uniform_case(4);
    c.ratio.assign(4, 2.0);
    c.length = {3.0, 2.0, 2.0000001, 5.0};
    Params p;
    p.min_element_length = 1.0;  // children must be longer than 1, so parents longer than 2
    p.max_refine_fraction = 1.0;
    p.theta = 1.0;
    const auto m = run(c, p);
    CHECK(m.candidates == std::vector<Index>{0, 2, 3});

    c.length = {2.0, 1.0, 0.5, 2.0};
    CHECK(run(c, p).status == Status::floor_exhausted);
    p.min_element_length = 0.0;  // disabled
    CHECK(run(c, p).candidates.size() == 4);
}

TEST_CASE("Equal errors keep their index order", "[marking]") {
    Case c = uniform_case(6);
    c.ratio.assign(6, 2.0);
    c.error = {1.0, 3.0, 1.0, 3.0, 2.0, 1.0};
    Params p;
    p.theta = 1.0;
    p.max_refine_fraction = 1.0;
    CHECK(run(c, p).marked == std::vector<Index>{1, 3, 4, 0, 2, 5});
}

TEST_CASE("Ties stay in index order on large inputs too", "[marking]") {
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> level(1, 3);
    Case c = uniform_case(300);
    for (auto& e : c.error)
        e = level(rng);
    c.ratio.assign(300, 2.0);
    Params p;
    p.theta = 1.0;
    p.max_refine_fraction = 1.0;
    std::vector<Index> expected(300);
    std::iota(expected.begin(), expected.end(), 0);
    std::sort(expected.begin(), expected.end(), [&](Index a, Index b) {
        const double ea = c.error[static_cast<std::size_t>(a)],
                     eb = c.error[static_cast<std::size_t>(b)];
        return ea != eb ? ea > eb : a < b;
    });
    CHECK(run(c, p).marked == expected);
}

TEST_CASE("The total is the whole mesh's, so unflagged error counts against theta", "[marking]") {
    Case c = uniform_case(4);
    c.error = {1.0, 1.0, 10.0, 10.0};
    c.ratio = {2.0, 2.0, 0.0, 0.0};  // the big errors are not flagged
    Params p;
    p.theta = 0.5;  // the candidates hold 2/202 of the total: the fallback
    p.max_refine_fraction = 1.0;
    CHECK(run(c, p).marked == std::vector<Index>{0, 1});
}

TEST_CASE("The per-cycle cap keeps the largest-error ceil(beta N)", "[marking]") {
    Case c = uniform_case(10);
    c.ratio.assign(10, 2.0);
    std::iota(c.error.begin(), c.error.end(), 1.0);
    Params p;
    p.theta = 1.0;
    p.max_refine_fraction = 0.25;  // ceil(2.5) = 3
    const auto m = run(c, p);
    CHECK(m.marked.size() == 10);
    CHECK(m.selected == std::vector<Index>{9, 8, 7});
}

TEST_CASE("The element ceiling admits the parents that fit, else stops", "[marking]") {
    Case c = uniform_case(10, 1.0, 7);  // T4: +7 each
    c.ratio.assign(10, 2.0);
    std::iota(c.error.begin(), c.error.end(), 1.0);
    Params p;
    p.theta = 1.0;
    p.max_refine_fraction = 1.0;
    p.max_elements = 10 + 3 * 7 + 6;  // room for floor(27 / 7) = 3 parents
    CHECK(run(c, p).selected == std::vector<Index>{9, 8, 7});
    p.max_elements = 10 + 3 * 7;  // exactly 3 parents: the ceiling itself is allowed
    CHECK(run(c, p).selected == std::vector<Index>{9, 8, 7});
    p.max_elements = 10 + 6;  // not even one
    const auto stopped = run(c, p);
    CHECK(stopped.status == Status::element_ceiling);
    CHECK(stopped.selected.empty());  // nothing was selected
    CHECK(stopped.marked.size() == 10);
    p.max_elements = 5;  // already above
    CHECK(run(c, p).status == Status::element_ceiling);
    p.max_elements = 10 + 7.5;  // fractional, as MATLAB allowed: one parent fits
    CHECK(run(c, p).selected == std::vector<Index>{9});
    p.max_elements = 0.5;  // on, and already exceeded
    CHECK(run(c, p).status == Status::element_ceiling);
    p.max_elements = -3;  // off
    CHECK(run(c, p).selected.size() == 10);
}

TEST_CASE("Parameters behave as the solver's did", "[marking]") {
    Case c = uniform_case(6);
    c.ratio.assign(6, 2.0);
    std::iota(c.error.begin(), c.error.end(), 1.0);
    Params p;
    p.max_refine_fraction = 1.0;
    p.theta = 1.2;  // unreachable: every candidate
    CHECK(run(c, p).marked.size() == 6);
    p.theta = 0.3;
    p.min_element_length = -1.0;  // off, like 0
    c.length.assign(6, 1e-9);
    CHECK(run(c, p).status == Status::selected);
    c.length.clear();  // lengths are not needed with the floor off
    CHECK(run(c, p).status == Status::selected);
}

TEST_CASE("Bad input is rejected", "[marking]") {
    Case c = uniform_case(3);
    Params p;
    p.theta = std::nan("");
    CHECK_THROWS_AS(run(c, p), std::invalid_argument);
    p = {};
    p.theta = -0.1;
    CHECK_THROWS_AS(run(c, p), std::invalid_argument);
    p = {};
    p.min_element_length = std::nan("");
    CHECK_THROWS_AS(run(c, p), std::invalid_argument);
    p = {};
    p.max_elements = std::numeric_limits<double>::infinity();
    CHECK_THROWS_AS(run(c, p), std::invalid_argument);
    p = {};
    p.max_refine_fraction = 0.0;
    CHECK_THROWS_AS(run(c, p), std::invalid_argument);
    p = {};
    c.error[1] = std::nan("");
    CHECK_THROWS_AS(run(c, p), std::invalid_argument);
    c = uniform_case(3);
    c.length.pop_back();
    p.min_element_length = 0.5;  // the floor needs every length
    CHECK_THROWS_AS(run(c, p), std::invalid_argument);
}
