# AMR — Parallel Adaptive Mesh Refinement

A linear (pointerless) adaptive mesh refinement engine built on a
**Morton/Z-order–encoded quadtree (2D) / octree (3D)**. Leaves are stored as a
sorted array of `(code, level)` pairs; refinement, 2:1 balancing, and
coarsening operate on that array. The design follows the p4est/t8code family
([BurstWG2011], [Holke2018]) and the non-conforming-AMR constraint framework
of [CDK2019]; see [REFERENCES.md](REFERENCES.md).

📖 **New here? Start with [docs/](docs/)** — [getting started](docs/getting-started.md),
how-tos, and design notes.

## Backends

Three implementations share the same algorithm and the same Morton core, each
in its own directory:

| Dir       | Backend            | Parallelism            | Build            | Tests                          |
|-----------|--------------------|------------------------|------------------|--------------------------------|
| `omp/`    | shared-memory C++  | OpenMP                 | CMake or `g++`   | Catch2 (`omp/tests.cpp`)       |
| `mpi/`    | distributed C++    | MPI (+OpenMP per rank) | `mpi/Makefile`   | Catch2 (`mpi/tests.cpp`)       |
| `cuda/`   | single-GPU         | CUDA + Thrust          | `cuda/Makefile`  | custom harness (`cuda/tests.cu`)|

Each C++ backend has the same module layout: `core` (Morton + strong types),
`tree` (refine/balance/coarsen), `physics` (oracles), `viz` (SVG/VTK), plus
`main` and `tests`.

## Invariants

- **Sorted leaves:** `leaf_codes` is ascending; `refine`/`coarsen` re-establish this.
- **Unique leaves:** no duplicate codes; no leaf is an ancestor of another.
- **2:1 balance:** after `balance()`, face-adjacent leaves differ by ≤ 1 level.
- **Partition of unity (MPI):** summed leaf volumes across ranks == 1.0
  (checked by `verify_global()`).

## Build & test

### OpenMP (`omp/`)

```bash
# Via CMake (fetches Catch2):
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
ctest --test-dir build

# Or directly against system Catch2, no network:
g++ -std=c++20 -O2 -fopenmp -Iomp omp/tests.cpp \
    /usr/lib/libCatch2Main.a /usr/lib/libCatch2.a -o omp/test && omp/test
OMP_NUM_THREADS=8 g++ -std=c++20 -O2 -fopenmp -Iomp omp/main.cpp -o omp/amr && omp/amr
```

### MPI (`mpi/`)

```bash
cd mpi && make            # needs mpicxx + system Catch2
mpirun -np 4 ./test
mpirun -np 4 ./amr
```

> In containers / sandboxes where the shared-memory transport stalls, force
> TCP: `mpirun --mca btl tcp,self -np 4 ./test`.

### CUDA (`cuda/`)

```bash
cd cuda && make ARCH=sm_80   # set ARCH to your GPU; default sm_70
./amr        # needs a working CUDA driver matching the runtime
```

> The Thrust device lambdas require nvcc's `--extended-lambda` (already in
> `cuda/Makefile`). The code compiles and links without a GPU; running needs a
> driver whose version matches the CUDA runtime.

## Benchmarks & docs

- `benchmarks/` — scripts, historical data, and the Colab notebook (see its
  README; new scaling runs are produced by the Stage-2 SLURM harness).
- `omp/PARALLELIZATION.md`, `cuda/WALKTHROUGH.md` — implementation notes
  imported from the earlier development line (marked historical where they
  describe superseded code).

## Status

`refine`, `balance`, and `coarsen` are verified across backends:

- OpenMP: 50038 assertions / 8 cases pass.
- MPI: full suite passes at 1/2/4 ranks (50038 assertions).
- CUDA: compiles and links clean (runtime needs a working GPU).

### OpenMP strong scaling (sample)

Fixed demo (`max_level=20`, ~89.3M leaves), `omp/` backend, 32-core host;
3D `balance()` wall-clock. Final leaf count is identical at every thread
count (parallel determinism). Full data: `benchmarks/results/omp_scaling_restructured_2026-05-29.csv`.

| threads | 3D balance (ms) | speedup |
|--------:|----------------:|--------:|
| 1       | 58267           | 1.0×    |
| 2       | 30305           | 1.9×    |
| 4       | 15634           | 3.7×    |
| 8       | 8365            | 7.0×    |
| 16      | 5927            | 9.8×    |
| 32      | 4701            | 12.4×   |

Proper strong/weak-scaling sweeps (parametrized sizes, MPI ranks, GPU) are
produced by the Stage-2 SLURM harness, not this fixed demo.
