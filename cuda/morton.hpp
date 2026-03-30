#pragma once
#include <cstdint>
#include <utility>
#include <tuple>

class Morton2D {
    static const uint64_t MASK_1 = 0x00000000FFFFFFFF;
    static const uint64_t MASK_2 = 0x0000FFFF0000FFFF;
    static const uint64_t MASK_3 = 0x00FF00FF00FF00FF;
    static const uint64_t MASK_4 = 0x0F0F0F0F0F0F0F0F;
    static const uint64_t MASK_5 = 0x3333333333333333;
    static const uint64_t MASK_6 = 0x5555555555555555;

public:
    static uint64_t spread(uint64_t n) {
        n &= MASK_1;
        n = (n | (n << 16)) & MASK_2;
        n = (n | (n << 8))  & MASK_3;
        n = (n | (n << 4))  & MASK_4;
        n = (n | (n << 2))  & MASK_5;
        n = (n | (n << 1))  & MASK_6;
        return n;
    }

    static uint64_t compact(uint64_t n) {
        n &= MASK_6;
        n = (n | (n >> 1)) & MASK_5;
        n = (n | (n >> 2)) & MASK_4;
        n = (n | (n >> 4)) & MASK_3;
        n = (n | (n >> 8)) & MASK_2;
        n = (n | (n >> 16)) & MASK_1;
        return n;
    }

    static uint64_t encode(uint32_t x, uint32_t y) {
        return (spread(y) << 1) | spread(x);
    }

    static std::pair<uint32_t, uint32_t> decode(uint64_t code) {
        return { static_cast<uint32_t>(compact(code)), static_cast<uint32_t>(compact(code >> 1)) };
    }

    // Returns UINT64_MAX if neighbor is invalid
    static uint64_t get_neighbor(uint64_t code, int level, int dx, int dy, int max_level = 21) {
        auto [x, y] = decode(code);
        uint64_t size = 1ULL << (max_level - level);
        
        // Use int64 for arithmetic to handle negative steps, but check bounds carefully
        int64_t nx = static_cast<int64_t>(x) + dx * static_cast<int64_t>(size);
        int64_t ny = static_cast<int64_t>(y) + dy * static_cast<int64_t>(size);
        int64_t limit = 1ULL << max_level;

        if (nx < 0 || nx >= limit || ny < 0 || ny >= limit) return UINT64_MAX;
        
        return encode(static_cast<uint32_t>(nx), static_cast<uint32_t>(ny));
    }
};

class Morton3D {
    static const uint64_t MASK_1 = 0x1FFFFF;
    static const uint64_t MASK_2 = 0x1F00000000FFFF;
    static const uint64_t MASK_3 = 0x1F0000FF0000FF;
    static const uint64_t MASK_4 = 0x100F00F00F00F00F;
    static const uint64_t MASK_5 = 0x10C30C30C30C30C3;
    static const uint64_t MASK_6 = 0x1249249249249249;

public:
    static uint64_t spread(uint64_t n) {
        n &= MASK_1;
        n = (n | (n << 32)) & MASK_2;
        n = (n | (n << 16)) & MASK_3;
        n = (n | (n << 8))  & MASK_4;
        n = (n | (n << 4))  & MASK_5;
        n = (n | (n << 2))  & MASK_6;
        return n;
    }

    static uint64_t compact(uint64_t n) {
        n &= MASK_6;
        n = (n | (n >> 2))  & MASK_5;
        n = (n | (n >> 4))  & MASK_4;
        n = (n | (n >> 8))  & MASK_3;
        n = (n | (n >> 16)) & MASK_2;
        n = (n | (n >> 32)) & MASK_1;
        return n;
    }

    static uint64_t encode(uint32_t x, uint32_t y, uint32_t z) {
        return (spread(z) << 2) | (spread(y) << 1) | spread(x);
    }

    static std::tuple<uint32_t, uint32_t, uint32_t> decode(uint64_t code) {
        return { 
            static_cast<uint32_t>(compact(code)), 
            static_cast<uint32_t>(compact(code >> 1)), 
            static_cast<uint32_t>(compact(code >> 2)) 
        };
    }

    static uint64_t get_neighbor(uint64_t code, int level, int dx, int dy, int dz, int max_level = 21) {
        auto [x, y, z] = decode(code);
        uint64_t size = 1ULL << (max_level - level);
        
        int64_t nx = static_cast<int64_t>(x) + dx * static_cast<int64_t>(size);
        int64_t ny = static_cast<int64_t>(y) + dy * static_cast<int64_t>(size);
        int64_t nz = static_cast<int64_t>(z) + dz * static_cast<int64_t>(size);
        int64_t limit = 1ULL << max_level;

        if (nx < 0 || nx >= limit || ny < 0 || ny >= limit || nz < 0 || nz >= limit) return UINT64_MAX;
        
        return encode(static_cast<uint32_t>(nx), static_cast<uint32_t>(ny), static_cast<uint32_t>(nz));
    }
};
