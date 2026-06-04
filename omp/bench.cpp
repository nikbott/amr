/**
 * @file bench.cpp
 * @brief Micro-benchmark: active-front balance() vs whole-mesh balance_ref().
 * @details Reports leaf counts, ripple passes, and wall time for the two balance
 * paths on the same fixtures. Build: see omp/Makefile (`make bench`) or
 *   g++ -O3 -fopenmp -std=c++20 -I. omp/bench.cpp -o omp_bench
 */
#include "tree.hpp"
#include "physics.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>

using namespace amr;

template <typename F>
double time_ms(F&& f) {
    auto s = std::chrono::high_resolution_clock::now();
    f();
    auto e = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(e - s).count();
}

// Fracture-like fixture: a thin fine crack (cells straddling x == mid) refined
// deep inside an otherwise coarse domain -> deep ripple over a large front.
template <int DIM>
struct CrackOracle {
    int max_lvl, fine;
    bool operator()(const Node& n, int) const {
        if (n.level >= fine) return false;
        std::array<Coordinate, DIM> c;
        if constexpr (DIM == 2) c = morton::decode_2d(n.code);
        else c = morton::decode_3d(n.code);
        uint64_t mid = (1ULL << max_lvl) / 2, size = 1ULL << (max_lvl - n.level);
        return (c[0].value <= mid && c[0].value + size > mid);
    }
};

template <int DIM, typename Oracle>
void run(const char* name, int max_lvl, Oracle oracle) {
    LinearTree<DIM> ref(max_lvl), act(max_lvl);
    while (ref.refine(oracle)) {}
    { Oracle o = oracle; while (act.refine(o)) {} }
    size_t pre = ref.size();

    double r = time_ms([&] { ref.balance_ref(); });
    double a = time_ms([&] { act.balance(); });

    std::cout << std::left << std::setw(8) << name << DIM << "D  leaves "
              << std::setw(9) << pre << " -> " << std::setw(9) << ref.size()
              << "  passes=" << std::setw(3) << ref.last_balance_iters
              << std::fixed << std::setprecision(1)
              << "  ref " << std::setw(8) << r << " ms  active " << std::setw(8) << a << " ms  "
              << std::setprecision(2) << "speedup " << (r / a) << "x\n";
}

int main() {
    std::cout << "=== omp balance(): active-front vs reference ===\n";
    std::cout << "threads=" << omp_get_max_threads() << "\n";

    Config cfg; cfg.max_level = 18; cfg.coarse_level = 3; cfg.fine_level = 9; cfg.radius = 0.25;
    run<2>("sphere", cfg.max_level, CircleOracle(cfg));
    run<3>("sphere", cfg.max_level, SphereOracle(cfg));
    run<2>("crack", 16, CrackOracle<2>{16, 16});
    run<3>("crack", 11, CrackOracle<3>{11, 11});
    return 0;
}
