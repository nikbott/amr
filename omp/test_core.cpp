/**
 * @file test_core.cpp
 * @brief Morton coding and neighbour codes (common/core.hpp and the trees' get_neighbor_code).
 */

#include <array>
#include <cstdint>
#include <random>
#include <vector>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "test_util.hpp"

using namespace amr;
using namespace amr::test;

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

TEST_CASE("get_neighbor_code returns the sentinel past the +domain boundary (core)",
          "[core][neighbor][boundary]") {
    // 3D at the maximum supported level: a coarse boundary cell's +y/+z neighbour
    // used to wrap to the origin (the carry ran off the top of the 64-bit word),
    // and +x used to land above every valid code, instead of the OOB sentinel.
    Octree tree(21);
    const uint64_t maxc = 1ULL << 21;
    const uint64_t step2 = 1ULL << (21 - 2);  // a level-2 cell's span per axis
    const uint64_t hi = maxc - step2;         // max aligned level-2 coordinate

    auto at = [&](uint64_t x, uint64_t y, uint64_t z) {
        Octree::Point p{Coordinate{static_cast<uint32_t>(x)},
                        Coordinate{static_cast<uint32_t>(y)},
                        Coordinate{static_cast<uint32_t>(z)}};
        return tree.encode(p);
    };

    SECTION("+x/+y/+z at the far corner are all out of bounds") {
        MortonCode corner = at(hi, hi, hi);
        REQUIRE(tree.get_neighbor_code(corner, 2, {1, 0, 0}).value == UINT64_MAX);
        REQUIRE(tree.get_neighbor_code(corner, 2, {0, 1, 0}).value == UINT64_MAX);
        REQUIRE(tree.get_neighbor_code(corner, 2, {0, 0, 1}).value == UINT64_MAX);
    }
    SECTION("interior +neighbours still resolve; -dir boundary is out of bounds") {
        MortonCode interior = at(step2, step2, step2);
        REQUIRE(tree.get_neighbor_code(interior, 2, {1, 0, 0}).value != UINT64_MAX);
        MortonCode origin = at(0, 0, 0);
        REQUIRE(tree.get_neighbor_code(origin, 2, {-1, 0, 0}).value == UINT64_MAX);
    }
    SECTION("a level-0 cell has no neighbour (and no 1<<64 shift UB)") {
        REQUIRE(tree.get_neighbor_code(MortonCode{0}, 0, {1, 0, 0}).value == UINT64_MAX);
        REQUIRE(tree.get_neighbor_code(MortonCode{0}, 0, {0, 1, 0}).value == UINT64_MAX);
    }
}

namespace {
/// morton::neighbor_code's oracle: decode, move one cell along each axis,
/// check the domain, encode.
template <int DIM>
uint64_t neighbor_by_coordinates(uint64_t code, int level, int max_level, const int* dir) {
    std::array<Coordinate, DIM> x;
    if constexpr (DIM == 2)
        x = morton::decode_2d(MortonCode{code});
    else
        x = morton::decode_3d(MortonCode{code});
    const int64_t side = int64_t{1} << (max_level - level);
    const int64_t extent = int64_t{1} << max_level;
    std::array<Coordinate, DIM> y;
    for (int k = 0; k < DIM; ++k) {
        const int64_t v = int64_t{x[static_cast<std::size_t>(k)].value} + dir[k] * side;
        if (v < 0 || v >= extent)
            return UINT64_MAX;
        y[static_cast<std::size_t>(k)] = Coordinate{static_cast<uint32_t>(v)};
    }
    if constexpr (DIM == 2)
        return morton::encode_2d(y[0], y[1]).value;
    else
        return morton::encode_3d(y[0], y[1], y[2]).value;
}

template <int DIM>
void check_neighbor(uint64_t code, int level, int max_level, const int* dir) {
    const uint64_t got = morton::neighbor_code<DIM>(code, level, max_level, dir);
    const uint64_t want = neighbor_by_coordinates<DIM>(code, level, max_level, dir);
    if (got != want) {
        INFO("code " << code << " level " << level << " max_level " << max_level << " dir "
                     << dir[0] << "," << dir[1] << "," << (DIM == 3 ? dir[2] : 0));
        CHECK(got == want);
    }
}
}  // namespace

TEMPLATE_TEST_CASE_SIG("neighbor_code matches decode, move, encode (core)",
                       "[core][neighbor]",
                       ((int DIM), DIM),
                       2,
                       3) {
    std::vector<std::array<int, 3>> dirs;
    for (int a = -1; a <= 1; ++a)
        for (int b = -1; b <= 1; ++b)
            for (int c = (DIM == 3 ? -1 : 0); c <= (DIM == 3 ? 1 : 0); ++c)
                dirs.push_back({a, b, c});

    SECTION("every aligned cell of every level of a small domain, every direction") {
        const int max_level = DIM == 2 ? 5 : 3;
        for (int level = 0; level <= max_level; ++level) {
            const uint32_t side = 1u << (max_level - level);
            const uint32_t extent = 1u << max_level;
            for (uint32_t i = 0; i < extent; i += side)
                for (uint32_t j = 0; j < extent; j += side)
                    for (uint32_t k = 0; k < (DIM == 3 ? extent : 1u); k += side) {
                        const uint64_t code =
                            DIM == 2
                                ? morton::encode_2d(Coordinate{i}, Coordinate{j}).value
                                : morton::encode_3d(Coordinate{i}, Coordinate{j}, Coordinate{k})
                                      .value;
                        for (const auto& d : dirs)
                            check_neighbor<DIM>(code, level, max_level, d.data());
                    }
        }
    }
    SECTION("random cells of the deepest trees: 62 bits in 2D, 63 in 3D") {
        const int max_level = morton::max_level_limit(DIM);
        std::mt19937_64 rng(DIM);
        for (int n = 0; n < 20000; ++n) {
            const int level = static_cast<int>(rng() % static_cast<uint64_t>(max_level + 1));
            const uint64_t align = ~((uint64_t{1} << (max_level - level)) - 1);
            const uint64_t limit = (uint64_t{1} << max_level) - 1;
            std::array<Coordinate, 3> c{};
            for (auto& v : c)
                v = Coordinate{static_cast<uint32_t>(rng() & limit & align)};
            const uint64_t code = DIM == 2 ? morton::encode_2d(c[0], c[1]).value
                                           : morton::encode_3d(c[0], c[1], c[2]).value;
            check_neighbor<DIM>(code, level, max_level, dirs[rng() % dirs.size()].data());
        }
    }
}

TEMPLATE_TEST_CASE_SIG("neighbor_code rejects what it cannot answer (core)",
                       "[core][neighbor]",
                       ((int DIM), DIM),
                       2,
                       3) {
    // No cell's code is UINT64_MAX, so the sentinel cannot be confused with a
    // neighbour: every code of the deepest tree stays below 2^63.
    const int deepest = morton::max_level_limit(DIM);
    const uint32_t far = (1u << deepest) - 1;
    const uint64_t corner =
        DIM == 2 ? morton::encode_2d(Coordinate{far}, Coordinate{far}).value
                 : morton::encode_3d(Coordinate{far}, Coordinate{far}, Coordinate{far}).value;
    CHECK(corner < (uint64_t{1} << 63));
    const int west[3] = {-1, 0, 0};
    CHECK(morton::neighbor_code<DIM>(corner, deepest, deepest, west) ==
          neighbor_by_coordinates<DIM>(corner, deepest, deepest, west));
    CHECK(morton::neighbor_code<DIM>(corner, deepest, deepest, west) != UINT64_MAX);
    // Out-of-range levels and trees deeper than the limit get the sentinel.
    const int east[3] = {1, 0, 0};
    CHECK(morton::neighbor_code<DIM>(0, 4, 3, east) == UINT64_MAX);
    CHECK(morton::neighbor_code<DIM>(0, -1, 3, east) == UINT64_MAX);
    CHECK(morton::neighbor_code<DIM>(0, 1, deepest + 1, east) == UINT64_MAX);
    CHECK_THROWS_AS(LinearTree<DIM>(deepest + 1), std::invalid_argument);
    CHECK_THROWS_AS(LinearTree<DIM>(-1), std::invalid_argument);
    CHECK_NOTHROW(LinearTree<DIM>(deepest));
}
