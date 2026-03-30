import re
import matplotlib.pyplot as plt
import numpy as np

def parse_sequential(filepath):
    """Parse sequential benchmark result."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    time_match = re.search(r'Total execution time: (\d+) ms', content)
    if time_match:
        return int(time_match.group(1))
    return None

def parse_openmp_results(filepath):
    """Parse OpenMP results from multiple runs."""
    times = {}
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Find all thread sections
    sections = re.findall(r'### Threads: (\d+).*?Total execution time: (\d+) ms', content, re.DOTALL)
    for threads, time in sections:
        if int(threads) not in times:
            times[int(threads)] = []
        times[int(threads)].append(int(time))
    
    # Calculate means
    mean_times = {t: sum(vals)/len(vals) for t, vals in times.items()}
    return mean_times

def parse_cuda_result(filepath):
    """Parse CUDA benchmark result."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    time_match = re.search(r'Total execution time: (\d+) ms', content)
    if time_match:
        return int(time_match.group(1))
    return None

# Parse all results
seq_time = parse_sequential('thread_benchmark_results_20_12_seq.txt')
print(f"Sequential: {seq_time} ms")

# Parse OpenMP results from 3 runs
omp_files = [
    'openmp/thread_benchmark_results_20_12_0.txt',
    'openmp/thread_benchmark_results_20_12_1.txt',
    'openmp/thread_benchmark_results_20_12_2.txt'
]

from collections import defaultdict
all_omp_times = defaultdict(list)

for omp_file in omp_files:
    times = parse_openmp_results(omp_file)
    for t, val in times.items():
        all_omp_times[t].append(val)

# Calculate mean for each thread count
omp_threads = sorted(all_omp_times.keys())
omp_mean_times = [sum(all_omp_times[t])/len(all_omp_times[t]) for t in omp_threads]
print(f"OpenMP threads: {omp_threads}")
print(f"OpenMP times: {omp_mean_times}")

# Parse CUDA results
cuda_colab_time = parse_cuda_result('cpp/results_colab.txt')
cuda_gpuserver_time = parse_cuda_result('cpp/results_gpuserver.txt')
print(f"CUDA Colab (Tesla T4): {cuda_colab_time} ms")
print(f"CUDA GPU Server (RTX 2080 SUPER): {cuda_gpuserver_time} ms")

# Create comparison plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Absolute execution times
x_pos = np.arange(len(omp_threads) + 3)
labels = ['Sequential'] + [f'OpenMP\n{t}T' for t in omp_threads] + ['CUDA\nColab', 'CUDA\nGPU Server']
times = [seq_time] + omp_mean_times + [cuda_colab_time, cuda_gpuserver_time]
colors = ['#1f77b4'] + ['#ff7f0e']*len(omp_threads) + ['#2ca02c', '#d62728']

bars = ax1.bar(x_pos, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
ax1.set_title('AMR Execution Time Comparison (MaxLvl=20, FineLvl=12)', fontsize=14, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(labels, fontsize=10)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}ms',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# Plot 2: Speedup relative to sequential
speedups = [seq_time / t for t in times]
bars2 = ax2.bar(x_pos, speedups, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.axhline(y=1, color='gray', linestyle='--', linewidth=2, label='Sequential Baseline')
ax2.set_ylabel('Speedup (vs Sequential)', fontsize=12, fontweight='bold')
ax2.set_title('Speedup Comparison', fontsize=14, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(labels, fontsize=10)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.legend()

# Add value labels on bars
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}x',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('full_comparison.png', dpi=300, bbox_inches='tight')
print("\n✅ Plot saved to full_comparison.png")

# Print summary table
print("\n" + "="*70)
print("SUMMARY TABLE")
print("="*70)
print(f"{'Implementation':<25} | {'Time (ms)':<12} | {'Speedup':<10}")
print("-"*70)
print(f"{'Sequential':<25} | {seq_time:<12} | {1.0:<10.2f}")
for i, t in enumerate(omp_threads):
    print(f"{'OpenMP (' + str(t) + ' threads)':<25} | {omp_mean_times[i]:<12.1f} | {speedups[i+1]:<10.2f}")
print(f"{'CUDA Colab (Tesla T4)':<25} | {cuda_colab_time:<12} | {speedups[-2]:<10.2f}")
print(f"{'CUDA GPU Server (RTX 2080)':<25} | {cuda_gpuserver_time:<12} | {speedups[-1]:<10.2f}")
print("="*70)

# Additional insights
print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)
print(f"• Best OpenMP speedup: {max(speedups[1:len(omp_threads)+1]):.2f}x ({omp_threads[speedups[1:len(omp_threads)+1].index(max(speedups[1:len(omp_threads)+1]))]} threads)")
print(f"• CUDA Colab speedup: {speedups[-2]:.2f}x")
print(f"• CUDA GPU Server speedup: {speedups[-1]:.2f}x")
print(f"• GPU Server is {cuda_colab_time/cuda_gpuserver_time:.2f}x faster than Colab")
print(f"• Best GPU is {seq_time/min(cuda_colab_time, cuda_gpuserver_time):.2f}x faster than sequential")
print("="*70)
