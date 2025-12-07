import numpy as np
import math
import cupy as cp
from numba import cuda, uint64, float64

# =========================================================
# FAST BITWISE OPS (Corrected SWAR)
# =========================================================

# 64-bit Masks
MASK2_1 = uint64(0x00000000FFFFFFFF)
MASK2_2 = uint64(0x0000FFFF0000FFFF)
MASK2_3 = uint64(0x00FF00FF00FF00FF)
MASK2_4 = uint64(0x0F0F0F0F0F0F0F0F)
MASK2_5 = uint64(0x3333333333333333)
MASK2_6 = uint64(0x5555555555555555)

@cuda.jit(device=True)
def device_spread_2d(n):
    """Dilates 32-bit int to 64-bit (inserts 0s)."""
    n = uint64(n)
    n &= MASK2_1
    n = (n | (n << 16)) & MASK2_2
    n = (n | (n << 8))  & MASK2_3
    n = (n | (n << 4))  & MASK2_4
    n = (n | (n << 2))  & MASK2_5
    n = (n | (n << 1))  & MASK2_6
    return n

@cuda.jit(device=True)
def device_compact_2d(n):
    """Inverse of spread."""
    n = uint64(n)
    n &= MASK2_6
    n = (n | (n >> 1)) & MASK2_5
    n = (n | (n >> 2)) & MASK2_4
    n = (n | (n >> 4)) & MASK2_3
    n = (n | (n >> 8)) & MASK2_2
    n = (n | (n >> 16)) & MASK2_1
    return n

@cuda.jit(device=True)
def device_encode_2d(x, y):
    return (device_spread_2d(x) | (device_spread_2d(y) << 1))

@cuda.jit(device=True)
def device_decode_2d(code):
    x = device_compact_2d(code)
    y = device_compact_2d(code >> 1)
    return x, y

# --- 3D MASKS ---
MASK3_1 = uint64(0x1FFFFF)
MASK3_2 = uint64(0x1F00000000FFFF)
MASK3_3 = uint64(0x1F0000FF0000FF)
MASK3_4 = uint64(0x100F00F00F00F00F)
MASK3_5 = uint64(0x10C30C30C30C30C3)
MASK3_6 = uint64(0x1249249249249249)

@cuda.jit(device=True)
def device_spread_3d(n):
    n = uint64(n)
    n &= MASK3_1
    n = (n | (n << 32)) & MASK3_2
    n = (n | (n << 16)) & MASK3_3
    n = (n | (n << 8))  & MASK3_4
    n = (n | (n << 4))  & MASK3_5
    n = (n | (n << 2))  & MASK3_6
    return n

@cuda.jit(device=True)
def device_compact_3d(n):
    n = uint64(n)
    n &= MASK3_6
    n = (n | (n >> 2))  & MASK3_5
    n = (n | (n >> 4))  & MASK3_4
    n = (n | (n >> 8))  & MASK3_3
    n = (n | (n >> 16)) & MASK3_2
    n = (n | (n >> 32)) & MASK3_1
    return n

@cuda.jit(device=True)
def device_encode_3d(x, y, z):
    return (device_spread_3d(x) | (device_spread_3d(y) << 1) | (device_spread_3d(z) << 2))

@cuda.jit(device=True)
def device_decode_3d(code):
    x = device_compact_3d(code)
    y = device_compact_3d(code >> 1)
    z = device_compact_3d(code >> 2)
    return x, y, z

# =========================================================
# KERNELS
# =========================================================

@cuda.jit
def kernel_flag_oracle_2d(codes, levels, flags, N, max_level, cx, cy, radius, bandwidth, fine_level, coarse_level):
    idx = cuda.grid(1)
    if idx < N:
        code = codes[idx]; lvl = levels[idx]
        x, y = device_decode_2d(code)
        
        # [FIX 1] Cast 1 to uint64 before shift to prevent overflow at max_level >= 32
        size = float64(uint64(1) << (max_level - lvl))
        
        node_cx = float64(x) + size * 0.5
        node_cy = float64(y) + size * 0.5
        
        dist = math.sqrt((node_cx - cx)**2 + (node_cy - cy)**2)
        
        # [FIX 2] Use semi-diagonal extent for Box-Circle intersection
        extent = size * 0.70710678
        
        should_refine = (lvl < coarse_level) or (abs(dist - radius) < (bandwidth + extent) and lvl < fine_level)
        flags[idx] = 4 if should_refine else 1

@cuda.jit
def kernel_flag_oracle_3d(codes, levels, flags, N, max_level, cx, cy, cz, radius, bandwidth, fine_level, coarse_level):
    idx = cuda.grid(1)
    if idx < N:
        code = codes[idx]; lvl = levels[idx]
        x, y, z = device_decode_3d(code)
        
        # [FIX 1] Cast to uint64
        size = float64(uint64(1) << (max_level - lvl))
        
        node_cx = float64(x) + size * 0.5
        node_cy = float64(y) + size * 0.5
        node_cz = float64(z) + size * 0.5
        
        dist = math.sqrt((node_cx - cx)**2 + (node_cy - cy)**2 + (node_cz - cz)**2)
        
        # [FIX 2] 3D Extent
        extent = size * 0.8660254
        
        should_refine = (lvl < coarse_level) or (abs(dist - radius) < (bandwidth + extent) and lvl < fine_level)
        flags[idx] = 8 if should_refine else 1

@cuda.jit
def kernel_refine_2d(old_codes, old_levels, flags, offsets, new_codes, new_levels, N, max_level):
    idx = cuda.grid(1)
    if idx < N:
        flag = flags[idx]
        start_idx = offsets[idx]
        code = old_codes[idx]; lvl = old_levels[idx]
        if flag == 1:
            new_codes[start_idx] = code
            new_levels[start_idx] = lvl
        elif flag == 4:
            x, y = device_decode_2d(code)
            new_lvl = lvl + 1
            # [FIX 1] Safe shift
            step = uint64(1) << (max_level - new_lvl)
            
            new_codes[start_idx] = device_encode_2d(x, y)
            new_levels[start_idx] = new_lvl
            new_codes[start_idx+1] = device_encode_2d(x + step, y)
            new_levels[start_idx+1] = new_lvl
            new_codes[start_idx+2] = device_encode_2d(x, y + step)
            new_levels[start_idx+2] = new_lvl
            new_codes[start_idx+3] = device_encode_2d(x + step, y + step)
            new_levels[start_idx+3] = new_lvl

@cuda.jit
def kernel_refine_3d(old_codes, old_levels, flags, offsets, new_codes, new_levels, N, max_level):
    idx = cuda.grid(1)
    if idx < N:
        flag = flags[idx]
        start_idx = offsets[idx]
        code = old_codes[idx]; lvl = old_levels[idx]
        if flag == 1:
            new_codes[start_idx] = code
            new_levels[start_idx] = lvl
        elif flag == 8:
            x, y, z = device_decode_3d(code)
            new_lvl = lvl + 1
            # [FIX 1] Safe shift
            step = uint64(1) << (max_level - new_lvl)
            
            for i in range(8):
                idx_u = uint64(i)
                dx = (idx_u & 1) * step
                dy = ((idx_u >> 1) & 1) * step
                dz = ((idx_u >> 2) & 1) * step
                new_codes[start_idx+i] = device_encode_3d(x+dx, y+dy, z+dz)
                new_levels[start_idx+i] = new_lvl

# --- DRIVER ---

def parallel_refine_step(current_codes, current_levels, config):
    N = len(current_codes)
    MAX_LEVEL = config.max_level
    ndim = 3 if len(config.center) == 3 else 2
    
    if isinstance(current_codes, cp.ndarray):
        d_codes = cuda.as_cuda_array(current_codes)
        d_levels = cuda.as_cuda_array(current_levels)
    else:
        d_codes = cuda.to_device(current_codes)
        d_levels = cuda.to_device(current_levels)
        
    d_flags = cuda.device_array(N, dtype=np.int32)
    threads = 256
    blocks = (N + threads - 1) // threads
    
    if ndim == 2:
        cx, cy = config.get_int_center(2)
        rad = config.get_int_radius(); bw = config.get_int_bandwidth()
        kernel_flag_oracle_2d[blocks, threads](
            d_codes, d_levels, d_flags, N, MAX_LEVEL, float64(cx), float64(cy), float64(rad), float64(bw), config.fine_level, config.coarse_level
        )
    else:
        cx, cy, cz = config.get_int_center(3)
        rad = config.get_int_radius(); bw = config.get_int_bandwidth()
        kernel_flag_oracle_3d[blocks, threads](
            d_codes, d_levels, d_flags, N, MAX_LEVEL, float64(cx), float64(cy), float64(cz), float64(rad), float64(bw), config.fine_level, config.coarse_level
        )
    cuda.synchronize()
    
    cp_flags = cp.asarray(d_flags)
    cp_offsets = cp.zeros(N + 1, dtype=cp.int32)
    cp.cumsum(cp_flags, out=cp_offsets[1:])
    total_new = int(cp_offsets[-1])
    
    d_new_codes = cuda.device_array(total_new, dtype=np.uint64)
    d_new_levels = cuda.device_array(total_new, dtype=np.uint8)
    d_offsets = cuda.as_cuda_array(cp_offsets[:-1])
    
    if ndim == 2:
        kernel_refine_2d[blocks, threads](d_codes, d_levels, d_flags, d_offsets, d_new_codes, d_new_levels, N, MAX_LEVEL)
    else:
        kernel_refine_3d[blocks, threads](d_codes, d_levels, d_flags, d_offsets, d_new_codes, d_new_levels, N, MAX_LEVEL)
    cuda.synchronize()
    
    cp_new_codes = cp.asarray(d_new_codes)
    cp_new_levels = cp.asarray(d_new_levels)
    sort_idx = cp.argsort(cp_new_codes)
    
    return cp_new_codes[sort_idx], cp_new_levels[sort_idx]
