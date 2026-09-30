/**
 * @file bench.cpp
 * @brief Throughput of one solver cycle (adapt) on growing meshes.
 *
 * @details For 2D and 3D structured seeds of growing size, two warm-up cycles
 * refine towards a smooth error bump, so the timed mesh has hanging nodes as
 * the solver's do; then adapt (θ = 0.70, β = 0.30, the report's marking) is
 * timed on that mesh, best of three. Prints each stage's time and the input
 * elements processed per second. Nothing is checked: the tests do that.
 *
 *   amr_simplex_bench [max_elements]   (default 2e6; meshes up to that size)
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#include "adapt.hpp"
#include "structured.hpp"

using namespace amr::simplex;

namespace {

/// Error bump centred in the box: elements near the centre are flagged.
void bump(const Mesh& m,
          const Point& centre,
          double width,
          std::vector<double>& error,
          std::vector<double>& ratio) {
    const auto n = static_cast<std::size_t>(m.num_elements());
    error.resize(n);
    ratio.resize(n);
    for (Index e = 0; e < m.num_elements(); ++e) {
        Point c{0, 0, 0};
        const auto v = m.element(e);
        for (Index i : v)
            for (std::size_t k = 0; k < 3; ++k)
                c[k] += m.pos[static_cast<std::size_t>(i)][k] / static_cast<double>(v.size());
        double d2 = 0;
        for (std::size_t k = 0; k < 3; ++k)
            d2 += (c[k] - centre[k]) * (c[k] - centre[k]);
        const auto i = static_cast<std::size_t>(e);
        error[i] = std::exp(-d2 / (width * width));
        ratio[i] = error[i] > 0.05 ? 1.5 : 0.5;
    }
}

}  // namespace

int main(int argc, char** argv) {
    const double max_elements = argc > 1 ? std::atof(argv[1]) : 2e6;
    AdaptParams p;
    p.marking.theta = 0.70;
    p.marking.max_refine_fraction = 0.30;
    std::printf("%3s %10s %10s %10s %10s %10s %10s %12s\n",
                "dim",
                "elements",
                "refined",
                "mark_ms",
                "balance_ms",
                "refine_ms",
                "total_ms",
                "Melem/s");
    for (int dim : {2, 3}) {
        const std::vector<double> dims =
            dim == 2 ? std::vector<double>{1008, 1016} : std::vector<double>{554, 568, 555};
        const Point centre = dim == 2 ? Point{504, 508, 0} : Point{277, 284, 277};
        for (Index n = dim == 2 ? 32 : 8;; n *= 2) {
            Mesh m = structured(dims,
                                std::vector<Index>(static_cast<std::size_t>(dim), n),
                                std::vector<double>(static_cast<std::size_t>(dim), 0.0));
            if (static_cast<double>(m.num_elements()) > max_elements / 4)
                break;
            std::vector<double> error, ratio;
            for (int warm = 0; warm < 2; ++warm) {
                bump(m, centre, dims[0] / 6, error, ratio);
                m = adapt(m, error, ratio, p).refinement.mesh;
            }
            bump(m, centre, dims[0] / 6, error, ratio);
            std::array<double, 3> best{std::numeric_limits<double>::infinity(),
                                       std::numeric_limits<double>::infinity(),
                                       std::numeric_limits<double>::infinity()};
            Index refined = 0;
            for (int rep = 0; rep < 3; ++rep) {
                const auto a = adapt(m, error, ratio, p);
                if (a.seconds[0] + a.seconds[1] + a.seconds[2] < best[0] + best[1] + best[2])
                    best = a.seconds;
                refined = a.refinement.mesh.num_elements();
            }
            const double total = best[0] + best[1] + best[2];
            std::printf("%3d %10d %10d %10.2f %10.2f %10.2f %10.2f %12.2f\n",
                        dim,
                        m.num_elements(),
                        refined,
                        1e3 * best[0],
                        1e3 * best[1],
                        1e3 * best[2],
                        1e3 * total,
                        static_cast<double>(m.num_elements()) / total / 1e6);
        }
    }
    return EXIT_SUCCESS;
}
