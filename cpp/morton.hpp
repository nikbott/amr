#pragma once
#include <cstdint>
#include <tuple>
#include <utility>
#include <immintrin.h> // For _pdep_u64 (BMI2)

class Morton2D {
    // Magic numbers for 2D bit interleaving (Spread 1 bit to 2 bits)
    static constexpr uint64_t MASK_1 = 0x00000000FFFFFFFF;
    static constexpr uint64_t MASK_2 = 0x0000FFFF0000FFFF;
    static constexpr uint64_t MASK_3 = 0x00FF00FF00FF00FF;
    static constexpr uint64_t MASK_4 = 0x0F0F0F0F0F0F0F0F;
    static constexpr uint64_t MASK_5 = 0x3333333333333333;
    static constexpr uint64_t MASK_6 = 0x5555555555555555;

    static constexpr int SHIFT_1 = 16;
    static constexpr int SHIFT_2 = 8;
    static constexpr int SHIFT_3 = 4;
    static constexpr int SHIFT_4 = 2;
    static constexpr int SHIFT_5 = 1;

public:
    static uint64_t spread(uint64_t n) {
        n &= MASK_1;
        n = (n | (n << SHIFT_1)) & MASK_2;
        n = (n | (n << SHIFT_2)) & MASK_3;
        n = (n | (n << SHIFT_3)) & MASK_4;
        n = (n | (n << SHIFT_4)) & MASK_5;
        n = (n | (n << SHIFT_5)) & MASK_6;
        return n;
    }

    static uint64_t compact(uint64_t n) {
        n &= MASK_6;
        n = (n | (n >> SHIFT_5)) & MASK_5;
        n = (n | (n >> SHIFT_4)) & MASK_4;
        n = (n | (n >> SHIFT_3)) & MASK_3;
        n = (n | (n >> SHIFT_2)) & MASK_2;
        n = (n | (n >> SHIFT_1)) & MASK_1;
        return n;
    }

    static uint64_t encode(uint32_t x, uint32_t y) {
#if defined(__BMI2__)
        // Hardware accelerated path
        // X goes to 0, 2, 4... (0x5555...)
        // Y goes to 1, 3, 5... (0xAAAA...)
        return _pdep_u64(x, 0x5555555555555555) | 
               _pdep_u64(y, 0xAAAAAAAAAAAAAAAA);
#else
        return (spread(y) << 1) | spread(x);
#endif
    }

    static std::pair<uint32_t, uint32_t> decode(uint64_t code) {
#if defined(__BMI2__)
        // Use PEXT (Parallel Bit Extract) if available
        // Note: compact() is the portable equivalent of PEXT
        return { 
            static_cast<uint32_t>(_pext_u64(code, 0x5555555555555555)), 
            static_cast<uint32_t>(_pext_u64(code, 0xAAAAAAAAAAAAAAAA)) 
        };
#else
        return { static_cast<uint32_t>(compact(code)), static_cast<uint32_t>(compact(code >> 1)) };
#endif
    }

    static uint64_t get_neighbor(uint64_t code, int level, int dx, int dy, int max_level) {
        auto [x, y] = decode(code);
        uint64_t size = 1ULL << (max_level - level);
        
        int64_t nx = static_cast<int64_t>(x) + dx * static_cast<int64_t>(size);
        int64_t ny = static_cast<int64_t>(y) + dy * static_cast<int64_t>(size);
        int64_t limit = 1ULL << max_level;

        if (nx < 0 || nx >= limit || ny < 0 || ny >= limit) return UINT64_MAX;
        
        return encode(static_cast<uint32_t>(nx), static_cast<uint32_t>(ny));
    }
};

class Morton3D {
    // Magic numbers for 3D bit interleaving (Spread 1 bit to 3 bits)
    static constexpr uint64_t MASK_1 = 0x1FFFFF;
    static constexpr uint64_t MASK_2 = 0x1F00000000FFFF;
    static constexpr uint64_t MASK_3 = 0x1F0000FF0000FF;
    static constexpr uint64_t MASK_4 = 0x100F00F00F00F00F;
    static constexpr uint64_t MASK_5 = 0x10C30C30C30C30C3;
    static constexpr uint64_t MASK_6 = 0x1249249249249249;

    static constexpr int SHIFT_1 = 32;
    static constexpr int SHIFT_2 = 16;
    static constexpr int SHIFT_3 = 8;
    static constexpr int SHIFT_4 = 4;
    static constexpr int SHIFT_5 = 2;

public:
    static uint64_t spread(uint64_t n) {
        n &= MASK_1;
        n = (n | (n << SHIFT_1)) & MASK_2;
        n = (n | (n << SHIFT_2)) & MASK_3;
        n = (n | (n << SHIFT_3)) & MASK_4;
        n = (n | (n << SHIFT_4)) & MASK_5;
        n = (n | (n << SHIFT_5)) & MASK_6;
        return n;
    }

    static uint64_t compact(uint64_t n) {
        n &= MASK_6;
        n = (n | (n >> SHIFT_5)) & MASK_5;
        n = (n | (n >> SHIFT_4)) & MASK_4;
        n = (n | (n >> SHIFT_3)) & MASK_3;
        n = (n | (n >> SHIFT_2)) & MASK_2;
        n = (n | (n >> SHIFT_1)) & MASK_1;
        return n;
    }

    static uint64_t encode(uint32_t x, uint32_t y, uint32_t z) {
#if defined(__BMI2__)
        // Hardware accelerated path (1 cycle on modern x86)
        // X: bits 0, 3, 6... -> Mask 0x9249... (1001 0010...)
        // Y: bits 1, 4, 7... -> Mask 0x2492... (0010 0100...)
        // Z: bits 2, 5, 8... -> Mask 0x4924... (0100 1001...)
        return _pdep_u64(z, 0x4924924924924924) | 
               _pdep_u64(y, 0x2492492492492492) | 
               _pdep_u64(x, 0x9249249249249249);
#else
        return (spread(z) << 2) | (spread(y) << 1) | spread(x);
#endif
    }

    static std::tuple<uint32_t, uint32_t, uint32_t> decode(uint64_t code) {
#if defined(__BMI2__)
        return { 
            static_cast<uint32_t>(_pext_u64(code, 0x9249249249249249)), 
            static_cast<uint32_t>(_pext_u64(code, 0x2492492492492492)), 
            static_cast<uint32_t>(_pext_u64(code, 0x4924924924924924)) 
        };
#else
        return { 
            static_cast<uint32_t>(compact(code)), 
            static_cast<uint32_t>(compact(code >> 1)), 
            static_cast<uint32_t>(compact(code >> 2)) 
        };
#endif
    }

    static uint64_t get_neighbor(uint64_t code, int level, int dx, int dy, int dz, int max_level) {
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