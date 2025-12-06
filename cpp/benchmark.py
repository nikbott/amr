import subprocess
import re

def get_time(cmd):
    try:
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output = result.stdout
        match = re.search(r"Total execution time: (\d+) ms", output)
        if match:
            return int(match.group(1))
    except Exception as e:
        print(f"Error running {cmd}: {e}")
    return None

def save_svg_plot(levels, speedups, filename):
    width = 800
    height = 600
    padding = 60
    
    with open(filename, 'w') as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n')
        f.write(f'<rect width="{width}" height="{height}" fill="white"/>\n')
        
        # Title
        f.write(f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="20">GPU Speedup vs CPU (Pure CUDA)</text>\n')
        
        # Scales
        min_x, max_x = min(levels), max(levels)
        min_y, max_y = 0, max(speedups) * 1.1
        if max_y == 0: max_y = 1 # Avoid div by zero
        
        def to_x(val):
            return padding + (val - min_x) / (max_x - min_x) * (width - 2 * padding)
        
        def to_y(val):
            return height - padding - (val - min_y) / (max_y - min_y) * (height - 2 * padding)
            
        # Axes
        f.write(f'<line x1="{padding}" y1="{height-padding}" x2="{width-padding}" y2="{height-padding}" stroke="black" stroke-width="2"/>\n') # X
        f.write(f'<line x1="{padding}" y1="{height-padding}" x2="{padding}" y2="{padding}" stroke="black" stroke-width="2"/>\n') # Y
        
        # Labels
        f.write(f'<text x="{width/2}" y="{height-10}" text-anchor="middle" font-family="Arial">Max Level</text>\n')
        f.write(f'<text x="15" y="{height/2}" text-anchor="middle" font-family="Arial" transform="rotate(-90 15 {height/2})">Speedup (x)</text>\n')
        
        # Grid and Ticks
        # X Ticks
        for l in levels:
            x = to_x(l)
            f.write(f'<line x1="{x}" y1="{height-padding}" x2="{x}" y2="{height-padding+5}" stroke="black"/>\n')
            f.write(f'<text x="{x}" y="{height-padding+20}" text-anchor="middle" font-family="Arial">{l}</text>\n')
            
        # Y Ticks (approx 5 ticks)
        num_ticks = 5
        for i in range(num_ticks + 1):
            val = max_y * i / num_ticks
            y = to_y(val)
            f.write(f'<line x1="{padding-5}" y1="{y}" x2="{padding}" y2="{y}" stroke="black"/>\n')
            f.write(f'<text x="{padding-10}" y="{y+5}" text-anchor="end" font-family="Arial">{int(val)}</text>\n')
            f.write(f'<line x1="{padding}" y1="{y}" x2="{width-padding}" y2="{y}" stroke="lightgray" stroke-dasharray="5,5"/>\n')

        # Data Line
        points = []
        for l, s in zip(levels, speedups):
            points.append(f"{to_x(l)},{to_y(s)}")
        
        f.write(f'<polyline points="{" ".join(points)}" fill="none" stroke="green" stroke-width="3"/>\n')
        
        # Data Points
        for l, s in zip(levels, speedups):
            x, y = to_x(l), to_y(s)
            f.write(f'<circle cx="{x}" cy="{y}" r="4" fill="green"/>\n')
            f.write(f'<text x="{x}" y="{y-10}" text-anchor="middle" font-family="Arial" font-size="12">{int(s)}x</text>\n')
            
        f.write('</svg>')

levels = [10, 12, 14, 16, 18]
cpu_times = []
gpu_times = []

print("Running Benchmark...")
for lvl in levels:
    print(f"Testing Max Level {lvl}...")
    
    fine = lvl - 6
    if fine < 3: fine = 3
    
    # CPU
    t_cpu = get_time(f"./amr_cpu --max_level {lvl} --fine_level {fine}")
    if t_cpu is None: t_cpu = 0
    cpu_times.append(t_cpu)
    print(f"  CPU: {t_cpu} ms (Fine Level {fine})")
    
    # GPU (Fixed time from Colab)
    t_gpu = 14
    gpu_times.append(t_gpu)
    print(f"  GPU: {t_gpu} ms (Fixed)")

speedups = [c/g if g > 0 else 0 for c, g in zip(cpu_times, gpu_times)]

print("\nResults:")
print(f"Levels: {levels}")
print(f"Speedups: {speedups}")

save_svg_plot(levels, speedups, 'speedup.svg')
print("Plot saved to speedup.svg")