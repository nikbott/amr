# Benchmarks

Benchmark scripts, data, and the Colab GPU notebook.

**Two vintages live here:**

- **Fresh (current layout).** `results/omp_scaling_restructured_2026-05-29.csv`
  — strong-scaling of the restructured `omp/` backend (see the table in the
  top-level [README](../README.md)). Produced by building `omp/main` per the
  README and sweeping `OMP_NUM_THREADS`.
- **Historical (`origin/main` line).** Everything else under `scripts/` and the
  other files in `results/`, imported from the development line that predates
  the `omp/ mpi/ cuda/` restructure.

**Caveat on the historical material.** The `scripts/` target the *old* flat
layout (`cpp/`, `openmp/`, loose root files) and binaries (`amr_seq`,
`amr_cpu`) that no longer exist; their `plot_*.py` read the original
`openmp/…`/`cpp/…` result paths. They are kept as reference — the empirical
data behind the early speedup figures and the GPU-server/Colab run recipe —
not as runnable tooling against the current tree.

**For new measurements,** build the backends as documented in the top-level
README; the parametrized strong/weak-scaling sweeps (sizes, MPI ranks, GPUs)
are produced by the Stage-2 SLURM harness, which supersedes these scripts.

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
