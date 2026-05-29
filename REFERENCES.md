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

**[Holke2018]** J. Holke. *Scalable algorithms for parallel tree-based adaptive
mesh refinement with general element types.* PhD thesis, Univ. of Bonn (2018).
Later: J. Holke et al., *t8code v1.0*, J. Open Source Softw. (2024).
Vendored at `~/Documents/ic/code/t8code/`.

Generalization of p4est to mixed element types via a "scheme" abstraction.
Reference implementation for the cross-backend parity tests; the algorithm
structure here (refine → repartition → balance) mirrors t8code's `t8_forest`
API. Used for validating correctness of new partitioning strategies.

**[CDK2019]** J. Červený, V. Dobrev, T. Kolev.
*Non-Conforming Mesh Refinement for High-Order Finite Elements.*
arXiv:1905.04033 (2019). LLNL-JRNL-751849.
PDF: `~/Documents/ic/refs/Non-Conforming Mesh Refinement For High-Order Finite Elements.pdf`.

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

See `code/REFERENCES.md` (`~/Documents/ic/code/REFERENCES.md`) for the parallel
DIC references: `[Sciuti2021]` (the MATLAB driver), `[ZZ1987]` (the error
estimator wrapped by the `ScalarFieldOracle`), `[HildRoux2006]`, `[Mathieu2015]`.

## Citation conventions

- In comments: `// [BurstWG2011, Alg. 2]` (C++) or `// [CDK2019] §4 — interpolation matrix`.
- In commit messages: `[CDK2019]` after the change description, e.g.
  `tree: align balance ripple with [CDK2019] Fig. 3`.
- When a citation is added, mention the file in PR description so reviewers can pull the PDF.
- The single source of truth is this file; code identifiers should not encode keys.
