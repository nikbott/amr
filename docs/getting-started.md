# Getting started

## Build and test (OpenMP)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release   # fetches Catch2
cmake --build build -j
ctest --test-dir build                            # run the unit tests
```

No network? Compile directly against system Catch2 — see the [README](../README.md#build--test).

## Run a demo

```bash
OMP_NUM_THREADS=8 ./build/amr     # refine + 2:1-balance a sphere, write SVG
```

The binary refines a geometric oracle, balances the tree, and writes an SVG of
the mesh. Tweak `dim`, `max_level`, and the oracle in `omp/main.cpp`.

## Other backends

```bash
cmake -S . -B build -DAMR_BUILD_MPI=ON -DAMR_BUILD_CUDA=ON   # enable as needed
```

- **MPI:** `mpirun -np 4 ./build/amr_mpi`
- **CUDA:** needs a GPU + matching driver; `./build/amr_cuda`

See the [README](../README.md) for the full build matrix and the
[Makefile](../Makefile) targets (`make build`, `make test`, `make ci-local`).

## What next

- Refine on your own criterion → [Add a refinement oracle](how-to-add-an-oracle.md).
- Understand the data layout and invariants → [README](../README.md).
