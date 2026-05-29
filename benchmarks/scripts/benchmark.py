import time
import pandas as pd
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning

from config import AMRConfig
from tree import Quadtree, Octree
from physics import CircleOracle2D, SphereOracle3D
from parallel import parallel_refine_step

def run_benchmark():
    if not cuda.is_available():
        print("Error: GPU not available.")
        return

    # Stress Test Configs (15 Million limit)
    cfg_2d = AMRConfig(max_level=32, fine_level=14, coarse_level=4, center=(0.5, 0.5), radius=0.25)
    cfg_3d = AMRConfig(max_level=21, fine_level=10, coarse_level=3, center=(0.5, 0.5, 0.5), radius=0.25)

    scenarios = [
        {"name": "2D Quadtree", "ndim": 2, "tree_cls": Quadtree, "oracle_cls": CircleOracle2D, "config": cfg_2d, "max_steps": 14, "limit": 15_000_000},
        {"name": "3D Octree", "ndim": 3, "tree_cls": Octree, "oracle_cls": SphereOracle3D, "config": cfg_3d, "max_steps": 10, "limit": 15_000_000}
    ]

    all_results = []
    print(f"{'Dim':<5} | {'Step':<5} | {'Elements':<10} | {'CPU (s)':<10} | {'GPU (s)':<10} | {'Speedup':<8}")
    print("-" * 65)

    for sc in scenarios:
        config = sc['config']
        seq_tree = sc['tree_cls'](max_level=config.max_level)
        oracle = sc['oracle_cls'](config)
        
        # GPU Start State (CuPy)
        par_codes = cp.array([0], dtype=np.uint64)
        par_levels = cp.array([0], dtype=np.uint8)

        # Warmup
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
            _ = parallel_refine_step(par_codes, par_levels, config)
        
        # Reset
        par_codes = cp.array([0], dtype=np.uint64)
        par_levels = cp.array([0], dtype=np.uint8)

        for step in range(sc['max_steps']): 
            n_prev = len(par_codes)

            # CPU
            if n_prev < 1_000_000:
                t0 = time.perf_counter()
                seq_tree.refine(oracle)
                t_seq = time.perf_counter() - t0
            else:
                t_seq = np.nan

            # GPU
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
                t0 = time.perf_counter()
                par_codes, par_levels = parallel_refine_step(par_codes, par_levels, config)
                cuda.synchronize()
                t_par = time.perf_counter() - t0
            
            n_gpu = len(par_codes)
            if t_par < 1e-6: t_par = 1e-6
            
            if not np.isnan(t_seq):
                speedup = t_seq / t_par
                all_results.append({"Dimension": sc['name'], "Elements": n_gpu, "Time (s)": t_seq, "Device": "CPU", "Speedup": 1.0})
            else:
                speedup = np.nan

            all_results.append({"Dimension": sc['name'], "Elements": n_gpu, "Time (s)": t_par, "Device": "GPU", "Speedup": speedup})
            
            cpu_str = f"{t_seq:<10.4f}" if not np.isnan(t_seq) else "Skipped"
            spd_str = f"{speedup:<8.2f}x" if not np.isnan(speedup) else "-"
            print(f"{sc['name']:<5} | {step:<5} | {n_gpu:<10} | {cpu_str} | {t_par:<10.4f} | {spd_str}")

            if n_gpu > sc['limit']: break
            
    df = pd.DataFrame(all_results)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    sns.lineplot(data=df, x="Elements", y="Time (s)", hue="Dimension", style="Device", markers=True, ax=ax1)
    ax1.set_xscale('log'); ax1.set_yscale('log'); ax1.set_title("Weak Scaling")
    df_gpu = df[(df["Device"] == "GPU") & (df["Speedup"].notna())]
    if not df_gpu.empty:
        sns.lineplot(data=df_gpu, x="Elements", y="Speedup", hue="Dimension", markers=True, ax=ax2)
        ax2.set_xscale('log'); ax2.set_title("GPU Acceleration")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_benchmark()
