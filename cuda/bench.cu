/**
 * @file bench.cu
 * @brief Micro-benchmark for the CUDA balance() ripple loop.
 * @details Reports leaf counts, ripple pass counts, and wall time so we can see
 * where balance() spends its time before/after the batched-ripple optimization.
 * Build: make ARCH=sm_86 bench ; run: ./bench
 */
#include "tree.cuh"
#include "physics.cuh"
#include <iostream>
#include <iomanip>

using namespace amr;

// Deep-ripple fixture: a single fine tower in a coarse domain -> many passes.
struct TowerOracle {
    int max_lvl; int dim; uint64_t mid;
    TowerOracle(int m, int d) : max_lvl(m), dim(d) { mid = (1ULL << m) / 2; }
    HOST_DEVICE bool operator()(MortonCode mc, int lvl) const {
        if (lvl >= max_lvl) return false;
        Coordinate x, y, z{0};
        if (dim == 2) morton::decode_2d(mc, x, y);
        else morton::decode_3d(mc, x, y, z);
        uint64_t size = 1ULL << (max_lvl - lvl);
        auto check = [&](uint64_t v) { return v <= mid && (v + size) > mid; };
        bool hit = check(x.value) && check(y.value);
        if (dim == 3) hit = hit && check(z.value);
        return hit;
    }
};

// Fracture-like fixture: a thin fine feature (line in 2D, plane in 3D) at the
// mid-coordinate refined to a deep level inside an otherwise coarse domain.
// Produces BOTH a deep ripple AND a large front -- the realistic DIC/crack case.
struct CrackOracle {
    int max_lvl; int dim; int fine; uint64_t mid;
    CrackOracle(int m, int d, int f) : max_lvl(m), dim(d), fine(f) { mid = (1ULL << m) / 2; }
    HOST_DEVICE bool operator()(MortonCode mc, int lvl) const {
        if (lvl >= fine) return false;
        Coordinate x, y, z{0};
        if (dim == 2) morton::decode_2d(mc, x, y);
        else morton::decode_3d(mc, x, y, z);
        uint64_t size = 1ULL << (max_lvl - lvl);
        // Refine any cell straddling the mid-plane x == mid (a vertical crack).
        return (x.value <= mid && (x.value + size) > mid);
    }
};

template <typename F>
float time_ms(F&& f) {
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s); f(); cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e); return ms;
}

template <int DIM, typename Oracle>
void bench_case(const char* name, int max_lvl, Oracle oracle) {
    // Two identical refined trees: one balanced the reference way, one active-front.
    LinearTree<DIM> ref(max_lvl), act(max_lvl);
    int steps = 0;
    while (ref.refine(oracle)) ++steps;
    { Oracle o = oracle; while (act.refine(o)); }
    size_t pre = ref.size();

    float ref_ms = time_ms([&]{ ref.balance_ref(); });
    float act_ms = time_ms([&]{ act.balance(); });

    std::cout << std::left << std::setw(8) << name << DIM << "D  "
              << "leaves " << std::setw(9) << pre << " -> " << std::setw(9) << ref.size() << "  "
              << "passes=" << std::setw(3) << ref.last_balance_iters << "  "
              << std::fixed << std::setprecision(3)
              << "ref " << std::setw(9) << ref_ms << " ms  "
              << "active " << std::setw(8) << act_ms << " ms  "
              << std::setprecision(2) << "speedup " << (ref_ms / act_ms) << "x\n";
}

int main() {
    std::cout << "=== balance() ripple benchmark ===\n";

    Config cfg; cfg.max_level = 20; cfg.coarse_level = 3; cfg.fine_level = 10; cfg.radius = 0.25;
    bench_case<2>("sphere", cfg.max_level, CircleOracle(cfg));
    bench_case<3>("sphere", cfg.max_level, SphereOracle(cfg));

    bench_case<2>("tower", 12, TowerOracle(12, 2));
    bench_case<3>("tower", 10, TowerOracle(10, 3));

    // Fracture-like: thin fine crack, deep level, large domain -> deep ripple at scale.
    bench_case<2>("crack", 16, CrackOracle(16, 2, 16));
    bench_case<3>("crack", 12, CrackOracle(12, 3, 12));
    return 0;
}
