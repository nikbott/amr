import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as mc
import cupy as cp
import numpy as np

# Import your existing modules
from config import AMRConfig
from parallel import parallel_refine_step
from morton import Morton2D, Morton3D

def decode_and_plot_2d(gpu_codes, gpu_levels, config: AMRConfig):
    """
    Downloads GPU data and plots the 2D Quadtree.
    """
    # 1. Transfer to CPU
    codes = cp.asnumpy(gpu_codes)
    levels = cp.asnumpy(gpu_levels)
    
    print(f"Plotting {len(codes)} elements...")

    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    rects = []
    
    # 2. Decode and Construct Geometry
    # We iterate on CPU because visualization is rarely the bottleneck compared to rendering
    for code, level in zip(codes, levels):
        # Use your Morton2D logic
        x, y = Morton2D.decode(int(code)) 
        
        # Calculate size based on your config logic
        # size = domain_width / 2^level
        size = 1 << (config.max_level - level)
        
        rects.append(patches.Rectangle((x, y), size, size))

    # 3. Fast Rendering using PatchCollection
    pc = mc.PatchCollection(rects, match_original=False, 
                            edgecolor='crimson', facecolor='none', linewidth=0.7)
    ax.add_collection(pc)
    
    # Set limits based on integer domain width
    limit = config.domain_width
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_aspect('equal')
    ax.set_title(f"Parallel 2D Mesh\nElements: {len(codes)} | Max Depth: {config.max_level}")
    plt.show()

def decode_and_plot_3d_slice(gpu_codes, gpu_levels, config: AMRConfig, slice_axis='z', slice_ratio=0.5):
    """
    Visualizing 3D Octrees is hard. This plots a 2D 'slice' through the 3D mesh.
    """
    codes = cp.asnumpy(gpu_codes)
    levels = cp.asnumpy(gpu_levels)
    
    print(f"Slicing 3D Mesh ({len(codes)} elements) at {slice_axis}={slice_ratio}...")

    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    rects = []
    
    # Calculate integer coordinate of the slice plane
    limit = config.domain_width
    slice_pos = int(limit * slice_ratio)
    
    count = 0
    for code, level in zip(codes, levels):
        x, y, z = Morton3D.decode(int(code))
        size = 1 << (config.max_level - level)
        
        # Check intersection with the slice plane
        # A cube intersects the plane if: pos <= slice < pos + size
        if slice_axis == 'z':
            if z <= slice_pos < z + size:
                rects.append(patches.Rectangle((x, y), size, size))
                count += 1
        elif slice_axis == 'y':
            if y <= slice_pos < y + size:
                rects.append(patches.Rectangle((x, z), size, size)) # Plot X-Z plane
                count += 1
        elif slice_axis == 'x':
            if x <= slice_pos < x + size:
                rects.append(patches.Rectangle((y, z), size, size)) # Plot Y-Z plane
                count += 1

    pc = mc.PatchCollection(rects, match_original=False, 
                            edgecolor='royalblue', facecolor='rgba(0,0,1,0.1)', linewidth=0.5)
    ax.add_collection(pc)
    
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_aspect('equal')
    ax.set_title(f"3D Slice ({slice_axis}={slice_ratio:.2f})\nIntersecting Elements: {count}/{len(codes)}")
    plt.show()

def main():
    # --- 1. Run 2D Parallel Simulation ---
    print("--- Running 2D GPU Simulation ---")
    cfg_2d = AMRConfig(max_level=16, fine_level=8, coarse_level=3)
    
    # Initialize on GPU
    codes = cp.array([0], dtype=np.uint64)
    levels = cp.array([0], dtype=np.uint8)
    
    # Step through refinement
    for _ in range(8): # Run enough steps to develop the mesh
        codes, levels = parallel_refine_step(codes, levels, cfg_2d)
    
    decode_and_plot_2d(codes, levels, cfg_2d)

    # --- 2. Run 3D Parallel Simulation ---
    print("\n--- Running 3D GPU Simulation ---")
    cfg_3d = AMRConfig(max_level=10, fine_level=6, coarse_level=2)
    
    codes = cp.array([0], dtype=np.uint64)
    levels = cp.array([0], dtype=np.uint8)
    
    for _ in range(6):
        codes, levels = parallel_refine_step(codes, levels, cfg_3d)
        
    # Visualize a slice through the middle (Z=0.5)
    decode_and_plot_3d_slice(codes, levels, cfg_3d, slice_axis='z', slice_ratio=0.5)

if __name__ == "__main__":
    main()
