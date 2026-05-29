import matplotlib.pyplot as plt
import numpy as np
import re
import os

def parse_time(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found!")
        return None
    with open(filepath, 'r') as f:
        content = f.read()
    # match "Total execution time: 123 ms"
    match = re.search(r'Total execution time:\s+(\d+)\s+ms', content)
    if match:
        return int(match.group(1))
    return None

# Paths
t4_path = 'results_colab.txt'
r2080_path = 'results_gpuserver.txt'

# Parse
t4_time = parse_time(t4_path)
r2080_time = parse_time(r2080_path)
seq_time = 19215.0 # Baseline from OpenMP Sequential (Reference)

print(f"Tesla T4 Time: {t4_time} ms")
print(f"RTX 2080 Time: {r2080_time} ms")

if t4_time is None or r2080_time is None:
    print("Error: Could not parse times. Using fallback/hardcoded values for plot generation if needed.")
    t4_time = 574 if t4_time is None else t4_time
    r2080_time = 387 if r2080_time is None else r2080_time

# Data preparation
gpus = ['Tesla T4\n(Google Colab)', 'RTX 2080 Super\n(GPU Server)']
times = [t4_time, r2080_time]
speedups = [seq_time / t for t in times]
colors = ['#2ca02c', '#d62728'] 

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Execution Time
bars1 = ax1.bar(gpus, times, color=colors, alpha=0.8, edgecolor='black', width=0.5)
ax1.set_ylabel('Tempo de Execução (ms)', fontsize=12, fontweight='bold')
ax1.set_title(f'Tempo de Execução (Menor é Melhor)\nRef: Sequencial = {int(seq_time)} ms', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Add labels
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)} ms',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# Plot 2: Speedup
bars2 = ax2.bar(gpus, speedups, color=colors, alpha=0.8, edgecolor='black', width=0.5)
ax2.set_ylabel('Speedup (vs CPU)', fontsize=12, fontweight='bold')
ax2.set_title('Speedup Relativo (Maior é Melhor)', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# Add labels
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}x',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('gpu_comparison.png', dpi=300)
print("Plot saved to gpu_comparison.png")
