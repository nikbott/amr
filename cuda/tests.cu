/**
 * @file tests.cu
 * @brief Complete Validation Suite for CUDA AMR (Literature Compliance).
 * @details Adapts all test cases from omp/tests.cpp.
 * CRITICAL: Uses independent geometric verification for 2:1 balance.
 */
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "physics.cuh"
#include "tree.cuh"

using namespace amr;

// --- Minimal Test Framework ---
#define CHECK(x)                                                        \
    if (!(x)) {                                                         \
        std::cerr << "FAIL: " << #x << " at line " << __LINE__ << "\n"; \
        exit(1);                                                        \
    }
#define REQUIRE(x) CHECK(x)
#define CHECK_NOTHROW(expr)                                                                      \
    try {                                                                                        \
        expr;                                                                                    \
    } catch (const std::exception& e) {                                                          \
        std::cerr << "FAIL: Unexpected exception '" << e.what() << "' for " << #expr             \
                  << " at line " << __LINE__ << "\n";                                            \
        exit(1);                                                                                 \
    } catch (...) {                                                                              \
        std::cerr << "FAIL: Unknown exception for " << #expr << " at line " << __LINE__ << "\n"; \
        exit(1);                                                                                 \
    }

// ==================================================================================
// HELPER: Robust Balance Verification (Geometric / Brute Force)
// Matches logic in omp/tests.cpp: count_balance_violations
// ==================================================================================
template <int DIM>
int count_balance_violations(const std::vector<uint64_t>& codes,
                             const std::vector<uint8_t>& levels,
                             int max_lvl) {
    int violations = 0;
    int n = codes.size();

    // Domain width in integer coords
    uint64_t width = 1ULL << max_lvl;

    // Geometric Directions -- full 2:1 balance: 6 face + 12 edge in 3D, so edge
    // violations are actually counted. 2D uses the first 4 (edge-complete).
    int dirs[18][3] = {{1, 0, 0},
                       {-1, 0, 0},
                       {0, 1, 0},
                       {0, -1, 0},
                       {0, 0, 1},
                       {0, 0, -1},
                       {1, 1, 0},
                       {1, -1, 0},
                       {-1, 1, 0},
                       {-1, -1, 0},
                       {1, 0, 1},
                       {1, 0, -1},
                       {-1, 0, 1},
                       {-1, 0, -1},
                       {0, 1, 1},
                       {0, 1, -1},
                       {0, -1, 1},
                       {0, -1, -1}};
    int num_dirs = (DIM == 2) ? 4 : 18;

    for (int i = 0; i < n; ++i) {
        uint64_t code = codes[i];
        int lvl = levels[i];
        uint64_t size = 1ULL << (max_lvl - lvl);

        // 1. Independent Decode
        Coordinate c[3] = {{0}, {0}, {0}};
        if constexpr (DIM == 2)
            morton::decode_2d(MortonCode{code}, c[0], c[1]);
        else
            morton::decode_3d(MortonCode{code}, c[0], c[1], c[2]);

        for (int d = 0; d < num_dirs; ++d) {
            // 2. Calculate Neighbor Coordinate Geometrically
            int64_t target_pos[3];
            bool out_of_bounds = false;

            for (int k = 0; k < DIM; ++k) {
                int64_t val = static_cast<int64_t>(c[k].value);
                if (dirs[d][k] == 1) {
                    // Right/Top: Adjacent is current + size
                    target_pos[k] = val + size;
                } else if (dirs[d][k] == -1) {
                    // Left/Bottom: Adjacent is current - 1
                    target_pos[k] = val - 1;
                } else {
                    target_pos[k] = val;
                }

                // Boundary Check
                if (target_pos[k] < 0 || target_pos[k] >= width) {
                    out_of_bounds = true;
                    break;
                }
            }
            if (out_of_bounds)
                continue;

            // 3. Re-encode Geometric Coordinate
            MortonCode target_mc;
            Coordinate t_coord[3];
            for (int k = 0; k < 3; ++k)
                t_coord[k].value = static_cast<uint32_t>(target_pos[k]);

            if constexpr (DIM == 2)
                target_mc = morton::encode_2d(t_coord[0], t_coord[1]);
            else
                target_mc = morton::encode_3d(t_coord[0], t_coord[1], t_coord[2]);

            // 4. Search for Leaf containing target_mc
            // std::upper_bound returns iterator to first element > value
            // The element containing target_mc must be immediately before upper_bound
            auto it = std::upper_bound(codes.begin(), codes.end(), target_mc.value);
            int idx = -1;

            if (it != codes.begin()) {
                idx = std::distance(codes.begin(), it) - 1;
            }

            if (idx != -1) {
                uint64_t found_code = codes[idx];
                int found_lvl = levels[idx];
                uint64_t found_size = 1ULL << ((max_lvl - found_lvl) * DIM);

                // Verify coverage (Partition of Unity check)
                if (found_code <= target_mc.value && (found_code + found_size) > target_mc.value) {
                    // 5. Check 2:1 Constraint
                    // Neighbor cannot be strictly coarser than (my_level - 1)
                    if (found_lvl < lvl - 1) {
                        violations++;
                    }
                }
            }
        }
    }
    return violations;
}

// ==================================================================================
// ORACLES
// ==================================================================================

struct L1Oracle {
    HOST_DEVICE bool operator()(MortonCode, int lvl) const { return lvl < 1; }
};

struct NoRefineOracle {
    HOST_DEVICE bool operator()(MortonCode, int) const { return false; }
};

struct RandomRefineOracle {
    HOST_DEVICE bool operator()(MortonCode mc, int lvl) const {
        if (lvl >= 4)
            return false;
        // Deterministic Hash
        uint64_t h = mc.value * 0x9e3779b97f4a7c15ULL;
        return (h % 3 == 0);
    }
};

struct GradedOracle {
    int max_lvl;
    int dim;
    HOST_DEVICE bool operator()(MortonCode mc, int lvl) const {
        if (lvl == 0)
            return true;
        if (lvl == 1) {
            uint64_t shift = (uint64_t)(max_lvl - 1) * dim;
            uint64_t mask = (1ULL << dim) - 1;
            uint64_t child_idx = (mc.value >> shift) & mask;
            return (child_idx == mask);
        }
        return false;
    }
};

struct TowerOracle {
    int max_lvl;
    int dim;
    uint64_t mid;
    TowerOracle(int m, int d) : max_lvl(m), dim(d) { mid = (1ULL << m) / 2; }

    HOST_DEVICE bool operator()(MortonCode mc, int lvl) const {
        if (lvl >= max_lvl)
            return false;
        Coordinate x, y, z{0};
        if (dim == 2)
            morton::decode_2d(mc, x, y);
        else
            morton::decode_3d(mc, x, y, z);

        uint64_t size = 1ULL << (max_lvl - lvl);
        auto check = [&](uint64_t val) { return val <= mid && (val + size) > mid; };

        bool hit = check(x.value) && check(y.value);
        if (dim == 3)
            hit = hit && check(z.value);
        return hit;
    }
};

struct CloudOracle {
    int fine_limit;
    HOST_DEVICE bool operator()(MortonCode mc, int lvl) const {
        if (lvl >= fine_limit)
            return false;
        return (mc.value % 7 == 0 || mc.value % 13 == 0);
    }
};

// ==================================================================================
// TEMPLATED TEST CASES
// ==================================================================================

template <int DIM>
void run_test_invariants() {
    std::cout << "[Test] Invariants " << DIM << "D... ";
    LinearTree<DIM> tree(5);

    RandomRefineOracle oracle;
    tree.refine(oracle);

    try {
        tree.verify();
    } catch (std::exception& e) {
        std::cerr << "Verify failed: " << e.what() << "\n";
        exit(1);
    }

    // Check neighbor integrity of bitwise logic vs geometric logic
    // We check the center node's +X neighbor
    uint64_t w = tree.domain_width();
    Coordinate center{static_cast<uint32_t>(w / 4)};
    MortonCode code;
    if constexpr (DIM == 2)
        code = morton::encode_2d(center, center);
    else
        code = morton::encode_3d(center, center, center);

    int dir[3] = {1, 0, 0};
    uint64_t n_code_val = get_neighbor_code<DIM>(code.value, 2, 5, dir);

    CHECK(n_code_val != UINT64_MAX);

    Coordinate nx, ny, nz;
    if constexpr (DIM == 2)
        morton::decode_2d(MortonCode{n_code_val}, nx, ny);
    else
        morton::decode_3d(MortonCode{n_code_val}, nx, ny, nz);

    uint64_t size_l2 = 1ULL << (5 - 2);
    CHECK(nx.value == center.value + size_l2);

    std::cout << "PASSED\n";
}

template <int DIM>
void run_test_geometric_coarsening() {
    std::cout << "[Test] Geometric Coarsening " << DIM << "D... ";
    LinearTree<DIM> tree(6);

    L1Oracle ref_oracle;
    while (tree.refine(ref_oracle))
        ;
    REQUIRE(tree.size() == (1ULL << DIM));

    NoRefineOracle crs_oracle;
    bool changed = tree.coarsen(crs_oracle);

    CHECK(changed);
    CHECK(tree.size() == 1);
    thrust::host_vector<uint8_t> h_lvl = tree.levels;
    CHECK(h_lvl[0] == 0);
    std::cout << "PASSED\n";
}

template <int DIM>
void run_test_graded_coarsening() {
    std::cout << "[Test] Graded Coarsening " << DIM << "D... ";
    int max_lvl = 5;
    LinearTree<DIM> tree(max_lvl);

    GradedOracle g_oracle{max_lvl, DIM};
    while (tree.refine(g_oracle))
        ;

    size_t expected = ((1ULL << DIM) - 1) + (1ULL << DIM);
    REQUIRE(tree.size() == expected);
    tree.verify();

    NoRefineOracle crs_oracle;
    while (tree.coarsen(crs_oracle))
        ;

    CHECK(tree.size() == 1);
    std::cout << "PASSED\n";
}

template <int DIM>
void run_test_deep_ripple() {
    std::cout << "[Test] Deep Ripple (Tower) " << DIM << "D... ";
    int max_lvl = 6;
    LinearTree<DIM> tree(max_lvl);

    TowerOracle t_oracle(max_lvl, DIM);
    while (tree.refine(t_oracle))
        ;

    tree.balance();
    tree.verify();

    thrust::host_vector<uint64_t> h_codes = tree.codes;
    thrust::host_vector<uint8_t> h_levels = tree.levels;
    std::vector<uint64_t> stl_codes(h_codes.begin(), h_codes.end());
    std::vector<uint8_t> stl_levels(h_levels.begin(), h_levels.end());

    // Uses the new GEOMETRIC verification
    int v = count_balance_violations<DIM>(stl_codes, stl_levels, max_lvl);
    CHECK(v == 0);

    std::cout << "PASSED\n";
}

template <int DIM>
void run_test_random_cloud() {
    std::cout << "[Test] Random Cloud " << DIM << "D... ";
    int max_lvl = 8;
    LinearTree<DIM> tree(max_lvl);

    CloudOracle c_oracle{5};
    for (int i = 0; i < 8; ++i)
        tree.refine(c_oracle);

    tree.balance();
    tree.verify();

    thrust::host_vector<uint64_t> h_codes = tree.codes;
    thrust::host_vector<uint8_t> h_levels = tree.levels;
    std::vector<uint64_t> stl_codes(h_codes.begin(), h_codes.end());
    std::vector<uint8_t> stl_levels(h_levels.begin(), h_levels.end());

    int v = count_balance_violations<DIM>(stl_codes, stl_levels, max_lvl);
    CHECK(v == 0);
    std::cout << "PASSED (" << tree.size() << " leaves)\n";
}

// ==================================================================================
// ACTIVE-FRONT PARITY
// The active-front balance() must produce a tree byte-identical to the simple
// whole-mesh balance_ref(). We refine an identical tree two ways and diff the
// final (codes, levels). This is the correctness oracle for the headline GPU
// optimization (cuda/tree.cuh: balance vs balance_ref).
// ==================================================================================

template <int DIM, typename Oracle>
void check_balance_parity(const char* name, int max_lvl, Oracle oracle, int refine_steps) {
    LinearTree<DIM> ref(max_lvl);
    LinearTree<DIM> act(max_lvl);
    for (int i = 0; i < refine_steps; ++i) {
        ref.refine(oracle);
        act.refine(oracle);
    }

    ref.balance_ref();  // baseline: whole-mesh re-check every pass
    act.balance();      // headline: active-front re-check

    REQUIRE(ref.size() == act.size());

    thrust::host_vector<uint64_t> rc = ref.codes, ac = act.codes;
    thrust::host_vector<uint8_t> rl = ref.levels, al = act.levels;
    for (size_t i = 0; i < rc.size(); ++i) {
        CHECK(rc[i] == ac[i]);
        CHECK(rl[i] == al[i]);
    }
    // Independent geometric confirmation that the active-front tree is balanced.
    std::vector<uint64_t> sc(ac.begin(), ac.end());
    std::vector<uint8_t> sl(al.begin(), al.end());
    CHECK(count_balance_violations<DIM>(sc, sl, max_lvl) == 0);
    CHECK(ref.last_balance_iters == act.last_balance_iters);  // same pass count
    std::cout << "  [parity] " << name << " " << DIM << "D OK (" << act.size() << " leaves, "
              << act.last_balance_iters << " passes)\n";
}

template <int DIM>
void run_test_active_parity() {
    std::cout << "[Test] Active-front parity " << DIM << "D...\n";
    check_balance_parity<DIM>("tower", 6, TowerOracle(6, DIM), 6);
    check_balance_parity<DIM>("cloud", 8, CloudOracle{5}, 8);
    check_balance_parity<DIM>("random", 5, RandomRefineOracle{}, 5);
    check_balance_parity<DIM>("graded", 5, GradedOracle{5, DIM}, 5);
    std::cout << "[Test] Active-front parity " << DIM << "D... PASSED\n";
}

// ==================================================================================
// MAIN
// ==================================================================================

void test_sfc_bijection() {
    std::cout << "[Test] SFC Bijection... ";
    Coordinate x{0x1234}, y{0x5678}, z{0x9ABC};

    MortonCode c2 = morton::encode_2d(x, y);
    Coordinate dx, dy;
    morton::decode_2d(c2, dx, dy);
    CHECK(x.value == dx.value);
    CHECK(y.value == dy.value);

    MortonCode c3 = morton::encode_3d(x, y, z);
    Coordinate dz;
    morton::decode_3d(c3, dx, dy, dz);
    CHECK(x.value == dx.value);
    CHECK(z.value == dz.value);
    std::cout << "PASSED\n";
}

void test_safety() {
    std::cout << "[Test] Safety & Limits... ";
    Quadtree t2(5);
    CHECK(t2.size() == 1);
    CHECK_NOTHROW(Quadtree(20));
    CHECK_NOTHROW(Octree(20));
    std::cout << "PASSED\n";
}

int main() {
    std::cout << "=== CUDA AMR Validation Suite ===\n";
    test_sfc_bijection();

    run_test_invariants<2>();
    run_test_invariants<3>();

    run_test_geometric_coarsening<2>();
    run_test_geometric_coarsening<3>();

    run_test_graded_coarsening<2>();
    run_test_graded_coarsening<3>();

    run_test_deep_ripple<2>();
    run_test_deep_ripple<3>();

    run_test_random_cloud<2>();
    run_test_random_cloud<3>();

    run_test_active_parity<2>();
    run_test_active_parity<3>();

    test_safety();
    std::cout << "All tests passed.\n";
    return 0;
}
