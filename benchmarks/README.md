# Benchmarks

Historical benchmark scripts, data, and the Colab GPU notebook, imported
from the `origin/main` development line (the fork that predates the
`omp/ mpi/ cuda/` backend restructure).

**Provenance / caveat.** These target the *old* flat layout (`cpp/`,
`openmp/`, loose root files), not the current backend layout. They are
kept as reference — empirical measurements behind the early speedup
figures and the recipe used to run on a GPU server / Colab — not as
runnable tooling against the current tree. The Stage 2 SLURM + CMake
harness supersedes them for new measurements.

## Layout

- `scripts/` — benchmark drivers and plotting scripts.
  - `benchmark.py`, `openmp_benchmark.py`, `cpp_benchmark.py` — per the
    origin dir they came from (root / `openmp/` / `cpp/`).
  - `*.sh` — thread sweep, sequential, and CUDA run wrappers.
  - `plot_*.py` — speedup / comparison figure generators.
- `results/` — raw measured data.
  - `thread_benchmark_*` — OpenMP thread-count sweeps (`<max_level>_<...>`).
  - `results_colab.txt`, `results_gpuserver.txt` — CUDA runs.
- `cuda.ipynb` — Colab notebook used for the GPU runs.

See `omp/PARALLELIZATION.md` and `cuda/WALKTHROUGH.md` for the
implementation notes that accompanied this data.
