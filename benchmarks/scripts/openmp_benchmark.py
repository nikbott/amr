import subprocess
import re
import os
import multiprocessing

def get_time(cmd):
    try:
        # Run command and capture output
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output = result.stdout
        
        # Parse execution time
        match = re.search(r"Total execution time: (\d+) ms", output)
        if match:
            return int(match.group(1))
        else:
            print(f"Warning: Could not parse time from output of: {cmd}")
            # print(output) # Debug
    except Exception as e:
        print(f"Error running {cmd}: {e}")
    return None

def save_svg_plot(levels, speedups, filename):
    width = 800
    height = 600
    padding = 80
    
    with open(filename, 'w') as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n')
        f.write(f'<rect width="{width}" height="{height}" fill="white"/>\n')
        
        # Title
        f.write(f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="20" font-weight="bold">OpenMP Speedup vs Sequential</text>\n')
        
        # Scales
        min_x, max_x = min(levels), max(levels)
        max_speedup = max(speedups) if speedups else 1.0
        min_y, max_y = 0, max_speedup * 1.2
        if max_y == 0: max_y = 1 
        
        def to_x(val):
            if max_x == min_x: return width / 2
            return padding + (val - min_x) / (max_x - min_x) * (width - 2 * padding)
        
        def to_y(val):
            return height - padding - (val - min_y) / (max_y - min_y) * (height - 2 * padding)
            
        # Axes
        f.write(f'<line x1="{padding}" y1="{height-padding}" x2="{width-padding}" y2="{height-padding}" stroke="black" stroke-width="2"/>\n') # X
        f.write(f'<line x1="{padding}" y1="{height-padding}" x2="{padding}" y2="{padding}" stroke="black" stroke-width="2"/>\n') # Y
        
        # Labels
        f.write(f'<text x="{width/2}" y="{height-20}" text-anchor="middle" font-family="Arial" font-size="14">Max Level</text>\n')
        f.write(f'<text x="20" y="{height/2}" text-anchor="middle" font-family="Arial" font-size="14" transform="rotate(-90 20 {height/2})">Speedup (x)</text>\n')
        
        # Grid and Ticks
        # X Ticks
        for l in levels:
            x = to_x(l)
            f.write(f'<line x1="{x}" y1="{height-padding}" x2="{x}" y2="{height-padding+5}" stroke="black"/>\n')
            f.write(f'<text x="{x}" y="{height-padding+20}" text-anchor="middle" font-family="Arial">{l}</text>\n')
            
        # Y Ticks
        num_ticks = 5
        for i in range(num_ticks + 1):
            val = max_y * i / num_ticks
            y = to_y(val)
            f.write(f'<line x1="{padding-5}" y1="{y}" x2="{padding}" y2="{y}" stroke="black"/>\n')
            f.write(f'<text x="{padding-10}" y="{y+5}" text-anchor="end" font-family="Arial">{val:.1f}</text>\n')
            f.write(f'<line x1="{padding}" y1="{y}" x2="{width-padding}" y2="{y}" stroke="lightgray" stroke-dasharray="5,5"/>\n')

        # Reference Line (1x Speedup)
        y_1x = to_y(1.0)
        if padding <= y_1x <= height - padding:
            f.write(f'<line x1="{padding}" y1="{y_1x}" x2="{width-padding}" y2="{y_1x}" stroke="red" stroke-dasharray="4,4" stroke-width="2"/>\n')
            f.write(f'<text x="{width-padding+5}" y="{y_1x+5}" font-family="Arial" font-size="12" fill="red">1x</text>\n')

        # Data Line
        if len(levels) > 1:
            points = []
            for l, s in zip(levels, speedups):
                points.append(f"{to_x(l)},{to_y(s)}")
            f.write(f'<polyline points="{" ".join(points)}" fill="none" stroke="#007bff" stroke-width="3"/>\n')
        
        # Data Points
        for l, s in zip(levels, speedups):
            x, y = to_x(l), to_y(s)
            f.write(f'<circle cx="{x}" cy="{y}" r="5" fill="#007bff"/>\n')
            f.write(f'<text x="{x}" y="{y-15}" text-anchor="middle" font-family="Arial" font-size="12" font-weight="bold">{s:.2f}x</text>\n')
            
        f.write('</svg>')

# Configuration
levels = [10, 15, 20]
cpu_times = []
par_times = []
num_threads = multiprocessing.cpu_count()

print(f"Running Benchmark with {num_threads} threads...")

for lvl in levels:
    print(f"\nTesting Max Level {lvl}...")
    
    # Adjust fine level to ensure significant work but not OOM
    # Heuristic: fine level is max_level - 6, clamped to min 3
    fine = lvl - 6
    if fine < 3: fine = 3
    
    # Sequential (1 Thread)
    cmd_seq = f"OMP_NUM_THREADS=1 ./build/amr --max_level {lvl} --fine_level {fine}"
    print(f"  Running Sequential: {cmd_seq}")
    t_seq = get_time(cmd_seq)
    if t_seq is None: t_seq = 0
    cpu_times.append(t_seq)
    print(f"  Sequential Time: {t_seq} ms")
    
    # Parallel (Max Threads)
    cmd_par = f"OMP_NUM_THREADS={num_threads} ./build/amr --max_level {lvl} --fine_level {fine}"
    print(f"  Running Parallel:   {cmd_par}")
    t_par = get_time(cmd_par)
    if t_par is None: t_par = 0
    par_times.append(t_par)
    print(f"  Parallel Time:   {t_par} ms")

# Calculate Speedups
speedups = []
for seq, par in zip(cpu_times, par_times):
    if par > 0:
        speedups.append(seq / par)
    else:
        speedups.append(0)

print("\nResults:")
print(f"Levels: {levels}")
print(f"Sequential Times: {cpu_times}")
print(f"Parallel Times:   {par_times}")
print(f"Speedups: {[f'{s:.2f}x' for s in speedups]}")

save_svg_plot(levels, speedups, 'speedup_comparison.svg')
print("\nPlot saved to speedup_comparison.svg")
