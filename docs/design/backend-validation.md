# Backend validation against SOTA (omp / mpi / cuda)

Validation of the `omp/`, `mpi/`, and `cuda/` backends against the
state-of-the-art parallel-AMR literature (p4est, Dendro-5.01, t8code), from a
verified deep-research pass (2026-06-03). Citation keys in
[`REFERENCES.md`](../REFERENCES.md). Companion: [`gpu-balance-notes.md`](gpu-balance-notes.md).
Status updated 2026-09-29 against the code; code is referenced by function,
not line, so the references survive edits.

## Verdict

**The overall design is validated as SOTA.** A linear Morton octree (sorted
leaf array + per-leaf level), SFC-based partitioning, 2:1 balance, and a
bisect/`lower_bound` reference oracle are exactly the design family of p4est
[BurstWG2011], Dendro-5.01, and t8code [Holke2018]. The gaps below are
*performance/scalability upgrades*, not correctness defects — our backends
produce correct, verified meshes today (cross-backend parity + geometric 2:1
checks pass).

Honesty notes from the research: (1) the strong claim that multi-round ripple
balance is "strictly inferior / wrong" was **refuted** — ripple is genuinely
correct and "inherently iterative" locally; the one-pass scheme is a *recommended
optimization*, not a fix. (2) The verified corpus was distributed/MPI-centric;
**OpenMP-specific** best-practice claims (scan, NUMA, false sharing) were *not*
independently evidenced, so omp/ items below lean on general HPC practice, not
cited AMR sources.

## omp/ (shared-memory) — `omp/tree.hpp`

| Aspect | Our code | SOTA | Action |
|---|---|---|---|
| Data structure | SoA linear octree, `Uninit<T>` to skip zero-init | matches p4est/t8code | none |
| Refinement scan | hand-rolled two-pass exclusive_scan (thread partials + serial fixup), `omp/core.hpp` | work-efficient; serial fixup is O(#threads), negligible | keep; optionally benchmark vs `std::exclusive_scan(std::execution::par_unseq)` |
| **2:1 balance** | **active front** (`balance()`): only cells next to a newly refined one are re-checked, with a full-check fallback on wide fronts; the full ripple stays as the oracle `balance_ref()` | the CUDA backend's front, ported | **done** (8769385); byte-identical to `balance_ref()` in the tests |
| Balance flag write | `#pragma omp atomic write flag=1` in `balance()` | monotonic, all writes identical → benign | none (the earlier "atomic-OR" concern is moot since every write is `1`) |
| Work scheduling | `schedule(dynamic,1024)` on the balance scan | reasonable for graded load imbalance | keep; watch false sharing on `wksp_flags` (byte array, dense writes) |

## mpi/ (distributed) — `mpi/tree.hpp`

| Aspect | Our code | SOTA | Action |
|---|---|---|---|
| Partition | equal-element SFC via `MPI_Exscan` partition map (`repartition()`) | p4est default is **equal-element**, computed without communication | **validated** — matches SOTA default |
| Partition (imbalance) | equal-count only | weighted partition via per-octant cost callback; **Hilbert** SFC (Dendro default) lowers ghost surface vs Morton | open: for crack work, an optional **weight/cost model** (helps only if deep octants cost more per element); Hilbert ordering to cut halo volume |
| **2:1 balance** | multi-round ripple: `update_partition_map()` + `fetch_ghosts()` **every pass** (`balance()`), `MPI_Allreduce` for convergence | p4est is **non-iterative**: local balance → **one** insulation-layer exchange → postbalance ([IBG2012]; insulation layer = 3^d envelope) | open, recommended: single-round insulation-layer balance. Not a correctness fix (ripple is correct), but removes per-pass communication. |
| **Ghost construction** | per-leaf neighbour enumeration over all local leaves; each needed node's owners found by binary search on the partition map, O(log P + k) (`fetch_ghosts()`, `find_owner_of_point()`) | recursive top-down `Search`, runtime ∝ *partition-boundary* leaves, **no 2:1 precondition** ([IBWG2015]; Holke/Knapp/Burstedde 2021) | **partly done**: the O(needed·P) rank scan is gone. Open: enumerate only partition-boundary leaves, so the work tracks the boundary rather than N_local. |
| **Ghost exchange** | **sparse**: non-blocking point-to-point when every rank has few partners, else `MPI_Neighbor_alltoallv` on a distributed-graph communicator (`exchange_data()`) | p4est: dedicated ghost object + **sparse symmetric non-blocking point-to-point** | **done** (d841ebb); moves the same bytes as the former dense `MPI_Alltoallv` |
| Empty partitions | all ranks reach collectives (the coarsen deadlock is fixed in `coarsen()`) | required invariant | **validated** — correct |

## Reference oracle (historical: `python/`, removed)

The pure-Python `Quadtree`/`Octree` formerly served as the parity oracle: its
balance predicate ("o,r unbalanced only if o ∈ I(r)") is exactly a `lower_bound`
insulation check, validated as a trustworthy ground truth. The Python backend
was **removed** (2026-06-05) once the in-tree C++ `balance_ref()` oracle (the
brute-force reference in `omp/tree.hpp` / `cuda/tree.cuh`) subsumed that role;
cross-backend parity now compares omp ↔ mpi ↔ cuda directly. The canonical AMR
invariants (sorted+unique, volume=1, 2:1, Morton round-trip, idempotent balance)
remain asserted in the C++ Catch2 suites.

## Prioritized action list

1. ~~**omp/ active-front balance**~~ — **done** (8769385).
2. ~~**mpi/ ghost exchange → sparse/neighborhood collectives**~~ — **done** (d841ebb), including the O(log P + k) owner lookup.
3. **mpi/ single-round insulation-layer balance** ([IBG2012]) — removes per-pass halo communication. *(Recommended, not correctness.)* Open.
4. **mpi/ ghost construction ∝ boundary** ([IBWG2015]) — enumerate only partition-boundary leaves. Open; pairs with #3.
5. **Weighted/Hilbert partitioning** for crack imbalance — only if a per-octant cost model shows deep octants are costlier; Hilbert lowers halo surface. Open.
6. ~~**Oracle invariant coverage**~~ — **done**: sorted and unique leaves, volume = 1, 2:1, Morton round-trip and idempotent balance are asserted in `omp/test_core.cpp` and `omp/test_tree.cpp`, and `verify()` checks levels, alignment and the domain (189ad7c).

## Caveats (from the verified pass)

- OpenMP-specific items (#1 aside) are general-practice, not cited-AMR-evidenced.
- "Ripple is obsolete" was **refuted**; treat balance upgrades as performance, not correctness.
- The Dendro 1.3×10¹² octant / 4 s benchmark is a historical self-reported figure (decommissioned Titan), scalability demonstration only.
- V&V specifics (exact tests p4est/Dendro/t8code ship) were not resolved — our cross-backend differential + geometric checks remain the practical ground truth.
