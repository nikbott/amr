/**
 * @file tests.cpp
 * @brief Validation Suite for Linear AMR System (Catch2 v3)
 * * @details
 * Validates the core invariants described in AMR literature:
 * 1. **Bijection**: Morton coding must be lossless and reversible.
 * 2. **Partition of Unity**: The sum of leaf volumes must exactly equal the domain volume.
 * 3. **2:1 Balance**: Verified using an independent geometric search (brute-force) to 
 * confirm the efficiency of the optimized Ripple algorithm.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <iostream>
#include <array>
#include <filesystem>

#include "tree.hpp"
#include "core.hpp"
#include "physics.hpp"
#include "../common/mesh_io.hpp"

using namespace amr;
using namespace Catch::Matchers;

// ==================================================================================
// HELPER: Robust Balance Verification (Public API Version)
// ==================================================================================

/**
 * @brief Counts 2:1 Balance Violations using independent geometric verification.
 */
template<int DIM>
int count_balance_violations(const LinearTree<DIM>& tree) {
    std::vector<std::array<int, DIM>> dirs;
    if constexpr (DIM == 2) dirs = {{1,0}, {-1,0}, {0,1}, {0,-1}};
    else dirs = {{1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1}};

    int violations = 0;
    int missing_neighbors = 0; // New: Track holes
    
    // Domain width in integer coordinates
    int64_t width = static_cast<int64_t>(tree.domain_width());

    for (const auto& node : tree) {
        // Independent decoding
        auto coords = tree.decode(node.code);
        int64_t size = 1LL << (tree.max_level - node.level);

        for (const auto& dir : dirs) {
            // 1. Calculate Expected Geometric Neighbor Coordinate
            std::array<int64_t, DIM> target_pos;
            bool out_of_bounds = false;

            for(int k=0; k<DIM; ++k) {
                int64_t current_val = static_cast<int64_t>(coords[k].value);
                if (dir[k] == 1) {
                    target_pos[k] = current_val + size;
                } else if (dir[k] == -1) {
                    target_pos[k] = current_val - 1; 
                } else {
                    target_pos[k] = current_val;
                }
                
                // Bounds Check
                if (target_pos[k] < 0 || target_pos[k] >= width) {
                    out_of_bounds = true;
                    break;
                }
            }
            if (out_of_bounds) continue;

            // 2. Re-encode to find the code covering this point
            typename LinearTree<DIM>::Point target_pt;
            for(int k=0; k<DIM; ++k) target_pt[k] = {static_cast<uint32_t>(target_pos[k])};
            
            MortonCode target_code = tree.encode(target_pt);

            // 3. Search for the leaf covering 'target_code'
            // We search for target_code. The covering leaf will be <= target_code.
            auto it = std::lower_bound(tree.begin(), tree.end(), Node{target_code, 0});
            
            Node neighbor{MortonCode{0}, 0};
            bool found = false;

            if (it != tree.end() && it->code == target_code) {
                // Exact match (rare, but possible if anchor aligns)
                neighbor = *it;
                found = true;
            } else if (it != tree.begin()) {
                // Check predecessor
                auto prev_it = it - 1;
                Node prev = *prev_it;
                
                uint64_t prev_size = 1ULL << (DIM * (tree.max_level - prev.level));
                if (prev.code.value <= target_code.value && 
                   (prev.code.value + prev_size) > target_code.value) {
                    neighbor = prev;
                    found = true;
                }
            }

            // 4. Check 2:1 constraint (STRICT)
            if (!found) {
                missing_neighbors++; // FAIL: Hole in domain
            } else {
                // If neighbor is strictly coarser than (node.level - 1), violation.
                if (neighbor.level < node.level - 1) {
                    violations++;
                }
            }
        }
    }
    return violations + missing_neighbors;
}

// ==================================================================================
// TEST SUITE
// ==================================================================================

TEST_CASE("Space-Filling Curve Bijection (Holke §3.1)", "[core][sfc]") {
    // Deterministic Random Generator
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint32_t> dist(0, std::numeric_limits<uint32_t>::max());

    SECTION("2D Morton Encoding (Strong Types)") {
        for (int i = 0; i < 10000; ++i) {
            // Mask to 31 bits
            Coordinate x{dist(rng) & 0x7FFFFFFF}; 
            Coordinate y{dist(rng) & 0x7FFFFFFF}; 
            
            MortonCode code = morton::encode_2d(x, y);
            auto [dx, dy] = morton::decode_2d(code);
            
            CHECK(x.value == dx.value);
            CHECK(y.value == dy.value);
        }
    }

    SECTION("3D Morton Encoding (Strong Types)") {
        for (int i = 0; i < 10000; ++i) {
            // Mask to 21 bits
            Coordinate x{dist(rng) & 0x1FFFFF}; 
            Coordinate y{dist(rng) & 0x1FFFFF};
            Coordinate z{dist(rng) & 0x1FFFFF};
            
            MortonCode code = morton::encode_3d(x, y, z);
            auto [dx, dy, dz] = morton::decode_3d(code);
            
            CHECK(x.value == dx.value);
            CHECK(y.value == dy.value);
            CHECK(z.value == dz.value);
        }
    }
}

TEMPLATE_TEST_CASE("Linear Tree Invariants (Burstedde §2.2)", "[tree]", Quadtree, Octree) {
    constexpr int DIM = (std::is_same<TestType, Quadtree>::value) ? 2 : 3;
    
    SECTION("Completeness & Partition of Unity") {
        int max_lvl = 5; 
        TestType tree(max_lvl);
        std::mt19937 rng(123);

        // Create a non-uniform random mesh
        tree.refine([&](const Node& n, int) { 
            return (n.level < 4 && (rng() % 3 == 0)); 
        });

        // 1. Sortedness
        REQUIRE(std::is_sorted(tree.begin(), tree.end()));
        
        // 2. Uniqueness (No Duplicate Morton Codes)
        auto it = std::adjacent_find(tree.begin(), tree.end());
        REQUIRE(it == tree.end());

        // 3. Partition of Unity
        double total_volume = 0.0;
        for (const auto& node : tree) {
            double side = 1.0 / (1ULL << node.level);
            total_volume += std::pow(side, DIM);
        }
        REQUIRE_THAT(total_volume, WithinAbs(1.0, 1e-9));

        // 4. Design-By-Contract Verification
        REQUIRE_NOTHROW(tree.verify());
    }

    SECTION("Neighbor Query Integrity") {
        TestType tree(4);
        // Uniform refinement to L2
        while(tree.refine([&](const Node& n, int){ return n.level < 2; }));

        // Check internal neighbor (Center of Domain)
        Coordinate size{static_cast<uint32_t>(tree.domain_width() / 4)}; 
        typename TestType::Point coords;
        for(int k=0; k<DIM; ++k) coords[k] = size;
        
        MortonCode code = tree.encode(coords);
        
        // Check Neighbor to the Right (+X)
        std::array<int, DIM> dir_right = {0}; dir_right[0] = 1; 
        
        MortonCode right_code = tree.get_neighbor_code(code, 2, dir_right);
        REQUIRE(right_code.value != UINT64_MAX); 
        
        auto right_coords = tree.decode(right_code);
        CHECK(right_coords[0].value == coords[0].value + size.value);
    }
}

TEMPLATE_TEST_CASE("Adaptivity & Coarsening (Burstedde §3.2)", "[amr][coarsen]", Quadtree, Octree) {
    constexpr int DIM = (std::is_same<TestType, Quadtree>::value) ? 2 : 3;

    SECTION("Geometric Coarsening") {
        TestType tree(6);
        // Refine Root -> L1 (Leaves = 2^DIM)
        tree.refine([&](const Node& n, int) { return n.level < 1; });
        REQUIRE(tree.size() == (1ULL << DIM));

        // Coarsen L1 -> Root
        bool changed = tree.coarsen([&](const Node&, int) { return false; }); 
        
        REQUIRE(changed);
        REQUIRE(tree.size() == 1);
        CHECK(tree[0].level == 0);
        REQUIRE_NOTHROW(tree.verify());
    }

    SECTION("Graded Coarsening (Misaligned Families)") {
        // Test robustness against misaligned families in the linear array.
        TestType tree(5);
        
        // Create Graded Mesh: Refine all to L1, then refine the LAST L1 node to L2.
        while(tree.refine([&](const Node& n, int) {
            if (n.level == 0) return true;
            if (n.level == 1) {
                // Find the last sibling (Code ends in 11...1)
                uint64_t last_sib_idx = (1ULL << DIM) - 1;
                uint64_t shift = DIM * (tree.max_level - 1);
                return ((n.code.value >> shift) & last_sib_idx) == last_sib_idx;
            }
            return false;
        }));

        // Expectation: 
        // L1 Nodes: (2^DIM - 1)
        // L2 Nodes: (2^DIM) (Refined from the last L1)
        size_t expected_size = ((1ULL << DIM) - 1) + (1ULL << DIM);
        REQUIRE(tree.size() == expected_size);
        REQUIRE_NOTHROW(tree.verify());

        // Force coarsening. 
        // The L2 family should merge. The misaligned L1 nodes should eventually merge.
        int passes = 0;
        while(tree.coarsen([&](const Node&, int) { return false; })) {
            passes++;
            tree.verify();
        }
        
        // Should return to root
        REQUIRE(tree.size() == 1);
        CHECK(tree[0].level == 0);
    }
}

TEMPLATE_TEST_CASE("2:1 Balance & Ripple Algorithm (Holke §3.3)", "[balance]", Quadtree, Octree) {
    constexpr int DIM = (std::is_same<TestType, Quadtree>::value) ? 2 : 3;
    
    SECTION("Deep Ripple (Cascade Propagation)") {
        // "Tower" Test: Refine center to Max Level.
        int max_lvl = 6; 
        TestType tree(max_lvl);
        
        auto center_oracle = [&](const Node& n, int) {
            auto coords = tree.decode(n.code);
            uint64_t size = 1ULL << (max_lvl - n.level);
            uint64_t mid = tree.domain_width() / 2;
            bool contains_center = true;
            for(int k=0; k<DIM; ++k) {
                if (!(coords[k].value <= mid && coords[k].value + size > mid)) contains_center = false;
            }
            return (contains_center && n.level < max_lvl);
        };
        
        while(tree.refine(center_oracle));
        
        // Execute Balance
        tree.balance();
        tree.verify();
        
        // Verification
        int violations = count_balance_violations(tree);
        REQUIRE(violations == 0);
    }

    SECTION("Random Cloud Stress Test") {
        TestType tree(8);
        auto cloud_oracle = [&](const Node& n, int) {
            if(n.level >= 5) return false;
            // Arbitrary math to create irregular shapes
            return (n.code.value % 7 == 0 || n.code.value % 13 == 0); 
        };
        
        for(int i=0; i<8; ++i) tree.refine(cloud_oracle);
        
        tree.balance();
        tree.verify();
        
        int violations = count_balance_violations(tree);
        REQUIRE(violations == 0);
    }
}

TEMPLATE_TEST_CASE("Active-front balance parity (byte-identical vs balance_ref)",
                   "[balance][parity]", Quadtree, Octree) {
    constexpr int DIM = (std::is_same<TestType, Quadtree>::value) ? 2 : 3;

    // Tree-independent oracles (decode the Node's own code) so two trees refine identically.
    auto decode = [](MortonCode code) {
        if constexpr (DIM == 2) return morton::decode_2d(code);
        else return morton::decode_3d(code);
    };

    auto run_parity = [&](const char* name, int max_lvl, auto oracle, int steps) {
        TestType ref(max_lvl), act(max_lvl);
        for (int i = 0; i < steps; ++i) { ref.refine(oracle); act.refine(oracle); }

        ref.balance_ref();   // whole-mesh baseline
        act.balance();       // active-front

        INFO("fixture=" << name << " DIM=" << DIM);
        REQUIRE(ref.size() == act.size());
        for (size_t i = 0; i < ref.size(); ++i) {
            REQUIRE(ref[i].code.value == act[i].code.value);
            REQUIRE(ref[i].level == act[i].level);
        }
        REQUIRE(count_balance_violations(act) == 0);   // independent geometric check
        REQUIRE(ref.last_balance_iters == act.last_balance_iters);
    };

    SECTION("Tower (deep ripple)") {
        int L = 6;
        run_parity("tower", L, [L, decode](const Node& n, int) {
            uint64_t mid = (1ULL << L) / 2, size = 1ULL << (L - n.level);
            auto c = decode(n.code);
            bool hit = true;
            for (int k = 0; k < DIM; ++k) if (!(c[k].value <= mid && c[k].value + size > mid)) hit = false;
            return hit && n.level < L;
        }, L);
    }
    SECTION("Random cloud") {
        run_parity("cloud", 8, [](const Node& n, int) {
            if (n.level >= 5) return false;
            return (n.code.value % 7 == 0 || n.code.value % 13 == 0);
        }, 8);
    }
    SECTION("Deterministic hash") {
        run_parity("random", 5, [](const Node& n, int) {
            if (n.level >= 4) return false;
            return (n.code.value * 0x9e3779b97f4a7c15ULL) % 3 == 0;
        }, 5);
    }
}

TEMPLATE_TEST_CASE("Binary mesh+field format round-trip (mesh_io C.2)",
                   "[mesh_io][integration]", Quadtree, Octree) {
    constexpr int DIM = (std::is_same<TestType, Quadtree>::value) ? 2 : 3;

    // Build a non-uniform mesh.
    int max_lvl = 6;
    TestType tree(max_lvl);
    std::mt19937_64 rng(7);
    for (int i = 0; i < 4; ++i)
        tree.refine([&](const Node& n, int) { return n.level < 5 && (rng() % 3 == 0); });

    // Pack into a MeshData with a non-trivial bbox + two fields (f64 and f32).
    mesh_io::MeshData m;
    m.dim = DIM;
    m.max_level = static_cast<uint32_t>(max_lvl);
    m.origin = {{1.5, -2.0, 3.25}};
    m.size   = {{100.0, 50.0, 12.5}};
    for (const auto& node : tree) {
        m.codes.push_back(node.code.value);
        m.levels.push_back(static_cast<uint8_t>(node.level));
    }
    mesh_io::Field err{"dic_error", true, {}};
    mesh_io::Field lvl{"level_f32", false, {}};
    for (size_t i = 0; i < m.codes.size(); ++i) {
        err.values.push_back(std::sin(static_cast<double>(i)) * 1.0e-3);
        lvl.values.push_back(static_cast<double>(m.levels[i]));
    }
    m.fields = {err, lvl};

    auto path = (std::filesystem::temp_directory_path() /
                 ("amr_mesh_io_" + std::to_string(DIM) + "d.bin")).string();

    SECTION("write/read preserves geometry, codes, levels, and fields") {
        mesh_io::write(path, m);
        mesh_io::MeshData r = mesh_io::read(path);

        REQUIRE(r.dim == m.dim);
        REQUIRE(r.max_level == m.max_level);
        for (uint32_t k = 0; k < m.dim; ++k) {
            REQUIRE(r.origin[k] == m.origin[k]);   // f64 bbox is exact
            REQUIRE(r.size[k]   == m.size[k]);
        }
        REQUIRE(r.codes == m.codes);               // bit-exact
        REQUIRE(r.levels == m.levels);

        REQUIRE(r.fields.size() == 2);
        REQUIRE(r.fields[0].name == "dic_error");
        REQUIRE(r.fields[0].f64);
        REQUIRE(r.fields[0].values == err.values);              // f64 exact
        REQUIRE(r.fields[1].name == "level_f32");
        REQUIRE_FALSE(r.fields[1].f64);
        for (size_t i = 0; i < lvl.values.size(); ++i)
            REQUIRE_THAT(r.fields[1].values[i], WithinAbs(lvl.values[i], 1e-5));  // f32 round-off
        std::filesystem::remove(path);
    }

    SECTION("rejects a corrupt magic") {
        { std::ofstream bad(path, std::ios::binary); bad << "XXXXnonsense"; }
        REQUIRE_THROWS_AS(mesh_io::read(path), std::runtime_error);
        std::filesystem::remove(path);
    }
}

TEST_CASE("Binary mesh+field format edge cases (mesh_io C.2)", "[mesh_io]") {
    auto path = (std::filesystem::temp_directory_path() / "amr_mesh_io_edge.bin").string();

    SECTION("empty mesh and field-less round-trip") {
        mesh_io::MeshData m; m.dim = 3; m.max_level = 10;
        mesh_io::write(path, m);
        mesh_io::MeshData r = mesh_io::read(path);
        REQUIRE(r.codes.empty());
        REQUIRE(r.fields.empty());
        REQUIRE(r.dim == 3);
        std::filesystem::remove(path);
    }

    SECTION("padding: n not a multiple of 8 round-trips with a trailing field") {
        mesh_io::MeshData m; m.dim = 2; m.max_level = 4;
        m.codes = {0, 1, 2, 3, 4};            // n = 5 -> 3 pad bytes before n_fields
        m.levels = {1, 1, 1, 1, 1};
        m.fields = {mesh_io::Field{"f", true, {0.1, 0.2, 0.3, 0.4, 0.5}}};
        mesh_io::write(path, m);
        mesh_io::MeshData r = mesh_io::read(path);
        REQUIRE(r.codes == m.codes);
        REQUIRE(r.fields.size() == 1);
        REQUIRE(r.fields[0].values == m.fields[0].values);
        std::filesystem::remove(path);
    }
}

TEST_CASE("Safety & Edge Cases", "[safety]") {
    // Only need one dimension type to test general logic logic
    using TestType = Quadtree;
    
    SECTION("Empty Tree Violation") {
        // This is tricky to test since the constructor guarantees a root node.
        // We would need to manually clear the private vectors, which we can't do.
        // Instead, we verify the constructor post-condition.
        TestType tree(5);
        REQUIRE(tree.size() > 0);
        REQUIRE_NOTHROW(tree.verify());
    }

    SECTION("Max Level Constraints") {
        // 2D Max is 31
        REQUIRE_NOTHROW(Quadtree(31)); 
        // 3D Max is 21
        REQUIRE_NOTHROW(Octree(21));
    }
}