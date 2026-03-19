# Adaptive Mesh Refinement (AMR)

![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)
![C++](https://img.shields.io/badge/language-C++-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-green.svg)
![OpenMP](https://img.shields.io/badge/OpenMP-Supported-blue.svg)

This repository contains a high-performance implementation of an **Adaptive Mesh Refinement (AMR)** system. The project explores the parallelization of AMR logic, transitioning from a sequential pointer-based approach to data-parallel architectures using multicore CPUs (OpenMP) and GPUs (CUDA). This repository serves as the foundation for research on parallelizing spatial data structures and is designed to support a scientific paper on the topic.

## 📌 Project Overview

Adaptive Mesh Refinement is a critical technique in scientific computing and computational physics, used to dynamically allocate computational resources (higher resolution mesh elements) only to regions that require higher accuracy (e.g., near boundaries, shocks, or areas of high gradient). 

In this repository, you'll find three distinct implementations of the AMR logic:
1. **Sequential CPU**: A traditional approach using a pointer-based dynamic data structure (`std::vector` or Trees).
2. **OpenMP Multicore CPU**: A concurrent implementation leveraging multi-threading to speed up refinement operations.
3. **CUDA GPU**: A highly optimized parallel stream processing model (Map-Scan-Scatter) designed for NVIDIA GPUs, avoiding pointer chasing and minimizing memory serialization bottlenecks.

## 📂 Repository Structure

The project is thoughtfully organized into specific modules based on hardware implementation and analysis tools:

- `sequential/` - Baseline CPU-only sequential implementation without parallel threading.
- `openmp/` - Multithreaded CPU implementation leveraging OpenMP for shared-memory parallelization.
- `cuda/` - GPU-accelerated implementation using NVIDIA's CUDA toolkit and a host-directed, device-executed pipeline.
- `scripts/` & `python/` - Utility scripts (mostly Python/bash) for benchmark orchestration, visualization, and performance plotting (e.g., speedup comparison graphs).
- `docs/` - Advanced documentation, including technical depth on parallelization strategies.
- `figures/` - Generated graphical outputs such as system architecture diagrams, SVG mesh visualizations, and generated benchmark plots.
- `results/` - Raw output logs and metrics from cluster/local benchmarks, used for plotting speedup and efficiency curves.

## 🚀 Key Technical Highlights (CUDA Implementation)

The most advanced module is our CUDA solver. Adapting AMR for GPUs inherently challenges the dynamically shifting nature of standard data structures, requiring a fundamental shift in algorithmic logic:
- **Map-Scan-Scatter Pipeline**: Resolves atomic write contention. Decouples the refinement evaluation logic from the memory allocation stage by utilizing Blelloch Scanning for parallel prefix sums.
- **GPU-Resident State**: Physics states remain completely on the device memory (VRAM) during refinement loops, minimizing PCIe bus data transfer overhead between the CPU and GPU.
- **Device-Side Bitonic Sort**: Maintains index continuity entirely in parallel, removing CPU bottlenecking.
- **Parallel Neighbor Search**: Fulfills the 2:1 balancing constraints efficiently through global static reads without locks.

*(For detailed information, please read `CUDA_PARALLELIZATION_GUIDE.md` and `CUDA_PRESENTATION.md` located in the repository.)*

## 🛠️ Building and Running

### Prerequisites
- A modern C++ compiler (GCC 9+ or Clang)
- CMake (version 3.10 or higher)
- NVIDIA CUDA Toolkit (if compiling the GPU model)
- OpenMP libraries (for CPU parallel version)
- Python 3.x (for running visualization scripts)

### Basic Build Example
Each specific directory naturally contains its own `Makefile` or `CMakeLists.txt` for independent building:

```bash
# E.g., Building the CUDA implementation
cd cuda
cmake .
make
./amr_cuda
```

*Note: You can inspect bash scripts like `run_experiments.sh` or `seq_bench.sh` within specific folders to run structured benchmarks.*

## 📊 Benchmarking & Visualizing

You can generate comprehensive speedup and execution performance charts using our provided python scripts:

```bash
cd scripts
python plot_speedup.py
```
Outputs and final mesh configurations are routinely saved to `figures/` and `results/` respectively. The program supports plotting refined quads out into interactive `.svg` layouts.

## 📝 License & Contact

This project is part of ongoing academic work in High-Performance Computing (HPC) research. Please refer to associated documentation or contact the authors if utilizing models for your own related research architectures.
