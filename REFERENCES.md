# References — `amr/` adaptive mesh refinement library

Stable bibliographic keys for citation in code comments. Reference like
`[CDK2019, §3]` or `[BurstWG2011, Alg. 2]`; never embed URLs or long titles
in source.

## Data structures and parallelism

**[BurstWG2011]** C. Burstedde, L. C. Wilcox, O. Ghattas.
*p4est: Scalable Algorithms for Parallel Adaptive Mesh Refinement on Forests of Octrees.*
SIAM J. Sci. Comput., **33**(3), 1103–1133 (2011).
DOI: [10.1137/100791634](https://doi.org/10.1137/100791634).

Foundational reference for the **linear (sorted) octree** data structure with
Morton (Z-order) encoding that this library implements. The 2:1 balance
algorithm, ghost-layer protocol, and Z-curve partitioning all derive from
this paper. Used as the design baseline for `omp/tree.hpp` and `mpi/tree.hpp`.

**[SSB2008]** H. Sundar, R. S. Sampath, G. Biros.
*Bottom-Up Construction and 2:1 Balance Refinement of Linear Octrees in Parallel.*
SIAM J. Sci. Comput., **30**(5), 2675–2708 (2008).
DOI: [10.1137/070681727](https://doi.org/10.1137/070681727).

The canonical parallel construction + **minimal 2:1 balance** of linear
Morton octrees via the *insulation-layer* property (no octant outside the 3^d
envelope can force a split). Output is the *coarsest* (minimal) balanced
octree; execution is iterative but bounded. Reference implementation = Dendro
(`github.com/paralab/Dendro-5.01`). See `docs/design/gpu-balance-notes.md` for how
the balance kernels here relate to it.

**[IBG2012]** T. Isaac, C. Burstedde, O. Ghattas.
*Low-Cost Parallel Algorithms for 2:1 Octree Balance.*
IEEE IPDPS 2012, 426–437. DOI: [10.1109/IPDPS.2012.47](https://doi.org/10.1109/IPDPS.2012.47).

The **GPU-amenable balance algorithm** we target: *octant preclusion* + the
`Reduce` step (collapse each family to one representative, |R| ≤ |S|/2^d), so
balance reduces to sort + binary-search + compaction + `Complete` instead of a
per-pass ripple. `Reduce` (Fig. 8), generate-coarse-neighbourhood + `Linearize`
(Fig. 6), and the preclusion variant (Fig. 7) are transcribed in
`docs/design/gpu-balance-notes.md`.

**[IBWG2015]** T. Isaac, C. Burstedde, L. C. Wilcox, O. Ghattas.
*Recursive Algorithms for Distributed Forests of Octrees.*
SIAM J. Sci. Comput., **37**(5), C497–C531 (2015).
DOI: [10.1137/140970963](https://doi.org/10.1137/140970963). arXiv:1406.0089.

Recursive **ghost/halo-layer construction** that works on arbitrarily refined
octrees (no 2:1 precondition) — the design reference for the `mpi/` ghost
exchange (Stage 2 tuning) and the multi-GPU halo path.

**[Holke2018]** J. Holke. *Scalable algorithms for parallel tree-based adaptive
mesh refinement with general element types.* PhD thesis, Univ. of Bonn (2018).
Later: J. Holke et al., *t8code v1.0*, J. Open Source Softw. (2024).
Vendored in the adaptive-dic repo at `t8code/`.

Generalization of p4est to mixed element types via a "scheme" abstraction.
Reference implementation for the cross-backend parity tests; the algorithm
structure here (refine → repartition → balance) mirrors t8code's `t8_forest`
API. Used for validating correctness of new partitioning strategies.

**[CDK2019]** J. Červený, V. Dobrev, T. Kolev.
*Non-Conforming Mesh Refinement for High-Order Finite Elements.*
arXiv:[1905.04033](https://arxiv.org/abs/1905.04033) (2019). LLNL-JRNL-751849.

Algebraic-constraint approach (variational restriction) for non-conforming
elements with hanging nodes — the foundation for the upcoming `common/oracle_scalar_field.hpp`
DIC-integration path. Their interpolation-matrix construction (§4) is the
template for `mesh_io`'s exported `[L]`/`S` matrix.

## Adaptive refinement strategy

**[Doerfler1996]** W. Dörfler. *A convergent adaptive algorithm for Poisson's equation.*
SIAM J. Numer. Anal., **33**(3), 1106–1124 (1996).

Bulk marking strategy used by `ScalarFieldOracle`: refine the smallest set of
leaves whose summed squared error covers a fraction θ of the total. Theta
defaults to 0.3 (a standard tradeoff between aggressiveness and stability).

**[BBHK2011]** W. Bangerth, C. Burstedde, T. Heister, M. Kronbichler.
*Algorithms and Data Structures for Massively Parallel Generic Adaptive Finite
Element Codes.* ACM Trans. Math. Softw., **38**(2), 14:1–28 (2011).

Reference for the deal.II + p4est coupling — used as the design pattern for
the MATLAB/MEX integration (the analog of `parallel::distributed::Triangulation`
is the `amr_handle_t` plus `amr_get_leaves` API to be added in Stage 5).

## Partitioning

**[Karypis_ParMETIS]** G. Karypis, V. Kumar.
*ParMETIS: Parallel Graph Partitioning and Sparse Matrix Ordering Library.*
University of Minnesota (1997 onwards).

Alternative partitioner for comparison studies in the AMR scaling paper.
Currently the library uses Z-curve partitioning (cheaper); ParMETIS is a
possible reviewer-requested baseline. Not a build-time dependency.

## Programming models

**[OpenMP5]** OpenMP Architecture Review Board. *OpenMP Application Programming
Interface, Version 5.2.* (Nov. 2021).

For `omp/*` annotations (parallel for, atomic, reduction, taskloop).

**[MPI4]** MPI Forum. *MPI: A Message-Passing Interface Standard, Version 4.1.* (2023).

For `mpi/*` collective operations (`MPI_Alltoallv`, neighborhood collectives).

**[CUDAGuide]** NVIDIA. *CUDA C++ Programming Guide* (current release).

For `cuda/*` kernels (cooperative groups, async copy, persistent kernels).

## DIC-side cross-reference

See the adaptive-dic `REFERENCES.md` for the parallel
DIC references: `[Sciuti2021]` (the MATLAB driver), `[ZZ1987]` (the error
estimator wrapped by the `ScalarFieldOracle`), `[HildRoux2006]`, `[Mathieu2015]`.

## Citation conventions

- In comments: `// [BurstWG2011, Alg. 2]` (C++) or `// [CDK2019] §4 — interpolation matrix`.
- In commit messages: `[CDK2019]` after the change description, e.g.
  `tree: align balance ripple with [CDK2019] Fig. 3`.
- When a citation is added, mention the file in PR description so reviewers can pull the PDF.
- The single source of truth is this file; code identifiers should not encode keys.
