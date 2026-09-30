/**
 * @file bench.cpp
 * @brief Throughput of one solver cycle (adapt) on growing meshes.
 *
 * @details For 2D and 3D structured seeds of growing size on the report's
 * domains, two warm-up cycles refine towards a smooth error bump, so the timed
 * mesh has hanging nodes as the solver's do. Then one call of adapt, with the
 * report's marking (θ = 0.70, β = 0.30), is timed whole, validation included,
 * best of three. The size floor sits, as in both of the report's modalities,
 * about three generations below the seed: just under the third, so the timed
 * call refines one generation past the warm-ups. (The report's absolute
 * floors, 3.5 px and 12.5 vx, would stop every seed finer than its own.)
 * Prints the call's wall time, its stages and the input elements processed
 * per second. Nothing is checked here: the tests do that.
 *
 *   amr_simplex_bench [max_elements]
 *
 * Seeds grow while their element count, 2 (n-1)^2 or 6 (n-1)^3, stays at or
 * below max_elements (default 1e6); the timed mesh is that seed after the two
 * warm-up cycles.
 */

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <string>
#include <vector>

#include "adapt.hpp"
#include "structured.hpp"

using namespace amr::simplex;

namespace {

/// Error bump centred in the box: elements near the centre are flagged.
void bump(const Mesh& m,
          const std::vector<double>& dims,
          std::vector<double>& error,
          std::vector<double>& ratio) {
    const auto n = static_cast<std::size_t>(m.num_elements());
    error.resize(n);
    ratio.resize(n);
    const double width = dims[0] / 6;
    for (Index e = 0; e < m.num_elements(); ++e) {
        Point c{0, 0, 0};
        const auto v = m.element(e);
        for (Index i : v)
            for (std::size_t k = 0; k < 3; ++k)
                c[k] += m.pos[static_cast<std::size_t>(i)][k] / static_cast<double>(v.size());
        double d2 = 0;
        for (std::size_t k = 0; k < dims.size(); ++k)
            d2 += (c[k] - dims[k] / 2) * (c[k] - dims[k] / 2);
        const auto i = static_cast<std::size_t>(e);
        error[i] = std::exp(-d2 / (width * width));
        ratio[i] = error[i] > 0.05 ? 1.5 : 0.5;
    }
}

bool refined(const Adaptation& a) {
    return a.status == AdaptStatus::refined || a.status == AdaptStatus::stagnated;
}

}  // namespace

int main(int argc, char** argv) try {
    const double max_elements = argc > 1 ? std::strtod(argv[1], nullptr) : 1e6;
    if (!(max_elements >= 1 && max_elements <= 1e8)) {
        std::fprintf(stderr, "usage: amr_simplex_bench [max_elements], 1 <= max_elements <= 1e8\n");
        return 2;
    }
    std::printf("%3s %10s %10s %10s %9s %9s %9s %9s %10s\n",
                "dim",
                "in_elems",
                "balanced",
                "out_elems",
                "call_ms",
                "mark_ms",
                "bal_ms",
                "ref_ms",
                "Melem/s");
    using Clock = std::chrono::steady_clock;
    for (int dim : {2, 3}) {
        const std::vector<double> dims =
            dim == 2 ? std::vector<double>{1008, 1016} : std::vector<double>{554, 568, 555};
        AdaptParams p;
        p.marking.theta = 0.70;
        p.marking.max_refine_fraction = 0.30;
        for (Index n = dim == 2 ? 32 : 8;; n *= 2) {
            const double seed = dim == 2 ? 2.0 * std::pow(n - 1, 2) : 6.0 * std::pow(n - 1, 3);
            if (seed > max_elements)
                break;
            Mesh m = structured(dims,
                                std::vector<Index>(static_cast<std::size_t>(dim), n),
                                std::vector<double>(static_cast<std::size_t>(dim), 0.0));
            // The seed's elements all have the same size L; its third
            // generation has L / 8.
            p.marking.min_element_length = 0.9 * std::pow(measure(m, 0), 1.0 / dim) / 8;
            std::vector<double> error, ratio;
            for (int warm = 0; warm < 2; ++warm) {
                bump(m, dims, error, ratio);
                auto a = adapt(m, error, ratio, p);
                if (refined(a))
                    m = std::move(a.refinement.mesh);
            }
            bump(m, dims, error, ratio);
            double best = std::numeric_limits<double>::infinity();
            Adaptation kept;
            for (int rep = 0; rep < 3; ++rep) {
                const auto t0 = Clock::now();
                auto a = adapt(m, error, ratio, p);
                const double wall = std::chrono::duration<double>(Clock::now() - t0).count();
                if (wall < best) {
                    best = wall;
                    kept = std::move(a);
                }
            }
            if (!refined(kept)) {
                std::printf("%3d %10d   (nothing to refine: status %d)\n",
                            dim,
                            m.num_elements(),
                            static_cast<int>(kept.status));
                continue;
            }
            std::printf("%3d %10d %10zu %10d %9.2f %9.2f %9.2f %9.2f %10.2f\n",
                        dim,
                        m.num_elements(),
                        kept.balanced.size(),
                        kept.refinement.mesh.num_elements(),
                        1e3 * best,
                        1e3 * kept.seconds[0],
                        1e3 * kept.seconds[1],
                        1e3 * kept.seconds[2],
                        static_cast<double>(m.num_elements()) / best / 1e6);
        }
    }
    return EXIT_SUCCESS;
} catch (const std::exception& e) {
    std::fprintf(stderr, "amr_simplex_bench: %s\n", e.what());
    return EXIT_FAILURE;
}
