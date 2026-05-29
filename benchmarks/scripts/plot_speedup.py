import re
import matplotlib.pyplot as plt
import os

def parse_time(filepath):
    """Parses total execution time from a benchmark file."""
    times = {}
    current_threads = None
    
    with open(filepath, 'r') as f:
        for line in f:
            # Check for thread count header
            thread_match = re.search(r'### Threads: (\d+)', line)
            if thread_match:
                current_threads = int(thread_match.group(1))
                continue
            
            # Check for Total execution time
            time_match = re.search(r'Total execution time: (\d+) ms', line)
            if time_match:
                time_ms = int(time_match.group(1))
                if current_threads is not None:
                    times[current_threads] = time_ms
                else:
                    # Assume single run/sequential if no thread header seen yet
                    # But for OpenMP file we expect headers. 
                    # For Seq file, return single value.
                    return time_ms
    return times

def main():
    seq_file = 'thread_benchmark_results_20_12_seq.txt'
    omp_files = [
        'openmp/thread_benchmark_results_20_12_0.txt',
        'openmp/thread_benchmark_results_20_12_1.txt',
        'openmp/thread_benchmark_results_20_12_2.txt'
    ]
    
    if not os.path.exists(seq_file):
        print(f"Error: {seq_file} not found.")
        return
    
    # 1. Get Sequential Baseline
    print(f"Reading Sequential results from {seq_file}...")
    seq_time = parse_time(seq_file)
    if isinstance(seq_time, dict):
        print("Error: Unexpected format for sequential file.")
        return
    print(f"Sequential Time: {seq_time} ms")

    # 2. Get OpenMP Results (Mean of 3 runs)
    from collections import defaultdict
    all_times = defaultdict(list)
    
    for omp_file in omp_files:
        if not os.path.exists(omp_file):
            print(f"Warning: {omp_file} not found. Skipping.")
            continue
            
        print(f"Reading OpenMP results from {omp_file}...")
        times = parse_time(omp_file)
        if not times:
            print(f"Warning: No results in {omp_file}.")
            continue
            
        for t, val in times.items():
            all_times[t].append(val)
    
    if not all_times:
        print("Error: No OpenMP results found.")
        return
        
    threads = sorted(all_times.keys())
    mean_times = []
    speedups = []
    
    for t in threads:
        avg_time = sum(all_times[t]) / len(all_times[t])
        mean_times.append(avg_time)
        speedups.append(seq_time / avg_time)
    
    # 3. Print Results Table
    print("\nBenchmark Results (Mean of 3 runs):")
    print(f"{'Threads':<10} | {'Mean Time':<10} | {'Speedup':<10}")
    print("-" * 36)
    print(f"{'Seq':<10} | {seq_time:<10} | 1.00x")
    for t, time, speedup in zip(threads, mean_times, speedups):
        print(f"{t:<10} | {time:<10.2f} | {speedup:.2f}x")

    # 4. Plot
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[!] matplotlib not found. Please install it to generate the plot:")
        print("    pip install matplotlib")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(threads, speedups, 'o-', linewidth=2, label='OpenMP Speedup (Mean)')
    
    # Ideal scaling
    plt.plot(threads, threads, 'k--', alpha=0.5, label='Ideal Linear Speedup')
    
    plt.xlabel('Number of Threads')
    plt.ylabel('Speedup (vs Sequential)')
    plt.title('AMR Scaling Comparison (OpenMP Mean vs Scalar Sequential)')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    plt.xscale('log', base=2) 
    plt.yscale('log', base=2)
    
    # Set ticks to match threads
    plt.xticks(threads, threads)
    y_ticks = [1, 2, 4, 8, 16, 32, 64]
    plt.yticks(y_ticks, y_ticks)
    
    output_file = 'speedup_comparison.png'
    plt.savefig(output_file)
    print(f"\nPlot saved to {output_file}")

if __name__ == "__main__":
    main()
