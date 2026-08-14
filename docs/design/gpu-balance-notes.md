# GPU 2:1 balance — algorithm notes and roadmap

Design notes for the CUDA `balance()` operation: what the literature says the
*ideal* algorithm is, why, and how our implementations map onto it. Citation
keys resolve in [`REFERENCES.md`](../REFERENCES.md).

## Problem and our context

We store a linear (sorted) Morton octree: leaves are axis-aligned cells encoded
as 64-bit Morton codes in a sorted array with a per-leaf level (p4est/t8code
style, [BurstWG2011]). The expensive operation is **2:1 balance**: enforce that
face-neighbouring leaves differ by at most one level. Our application (DIC for
fracture) produces *thin, deeply refined crack features in coarse domains*,
i.e. **deep ripple fronts over large meshes** — the adversarial regime for any
balancer.

## The algorithmic landscape

There are three paradigms, in increasing sophistication:

1. **Ripple** ([SSB2008] §I; what `cuda/tree.cuh::balance_ref` does). Mark every
   leaf whose face-neighbour is >1 level coarser, split the coarse violator one
   level, repeat until no violations. Correct and simple, but *"splitting one
   octant can force a non-adjacent octant to split"* ([SSB2008], the *ripple
   effect*), so it is inherently iterative and each pass re-scans O(N).

2. **Active-front ripple** (our production `cuda/tree.cuh::balance`). Same
   output, but each pass re-checks only the **insulation / preclusion layer** —
   the just-refined children plus the pre-existing finer cells that can still
   flag them — instead of the whole mesh. **Byte-identical** to `balance_ref`
   (test `run_test_active_parity`, incl. a wide-front `crack` fixture),
   ~2.2× faster on the sphere (89 M leaves) and ~1.8× on a thin crack. When that
   layer grows to span the mesh (a crack *plane*: one coarse cell fans out to
   the whole domain), a cheap pre-filter plus one O(n) reduce detect it and the
   balance **falls through, stickily, to the plain whole-mesh check** that
   `balance_ref` uses — zero front-tracking overhead — so `balance ≥ balance_ref`
   on every workload. This is an engineering approximation of the next idea.

3. **Insulation-layer + preclusion** ([SSB2008], [IBG2012]; the *ideal* target).
   The key theorem ([IBG2012] §II-B): two octants `o`, `r` can be unbalanced
   **only if `o` lies in `r`'s insulation layer `I(r)`** — the 3^d envelope of
   like-sized octants around `r`. Nothing outside `I(r)` can force `r` to split.
   This bounds the ripple and lets balance be done by *generating* the cells
   each leaf demands, then completing the octree — work proportional to the
   active boundary, not O(N) per pass.

> **Crucial distinction the research flagged:** "iterative in execution" ≠
> "non-minimal in output". [SSB2008] produces the **coarsest** (minimal)
> balanced octree. Do not conflate "not single-pass" with "not minimal".

## Definitions ([IBG2012], verbatim where quoted)

- **Insulation layer `I(r)`**: *"an envelope of 3^d like-sized octants"* around
  `r`. Balance information only needs to flow within insulation layers.
- **Coarse neighbourhood `N(o)`**: the parent-level octants neighbouring
  `parent(o)`. *"each l-octant ... attempt[s] to add to the octree its family
  and the (l+1)-octants that neighbor its parent ... the coarse neighborhood N"*.
  Which neighbours (face / edge / corner) depends on the balance condition
  ([IBG2012] Fig. 5). **Our `balance_ref`/`balance` enforce *face* balance**, so
  `N(o)` = the 2·d face-neighbours of `parent(o)` at the parent's level.
- **Preclusion `r ≺ o`** ("o precludes r"): *"if and only if parent(r) is
  ancestor (or equal) to parent(o)."* Equivalence classes are families;
  precluded octants are redundant and removed, then recovered by completion.
- **`0-sibling(o)`**: the first child of `parent(o)` — the canonical family
  representative.
- **`Reduce(S)`** ([IBG2012] Fig. 8): collapse each family in the sorted set to
  one representative. *"R is the smallest subset of leaves that can be used to
  reconstruct a linear octree using a completion algorithm"*; **|R| ≤ |S|/2^d**.
- **`Linearize`**: O(n) removal of coarse octants that overlap a finer one
  (keep only leaves). **`Complete`**: O(n) fill of gaps between leaves in the
  coarsest way.

## The algorithm boxes ([IBG2012])

**`Reduce(S)` (Fig. 8)** — sorted in, sorted out, ≤ |S|/2^d:

```
R[0] ← 0-sibling(S[0]); i ← 1
for 0 ≤ j < |S|:
    s ← 0-sibling(S[j])           # canonical rep of S[j]'s family
    r ← R[i-1]                    # last kept
    if r ≺ s:    R[i-1] ← s       # s is finer & same lineage → replace
    else if s ⊀ r: R[i] ← s; i++  # different family → append
```

**Old subtree balance (Fig. 6)** — correct, ripple-style, our reference:

```
S_new ← ∅
for all o ∈ S ∪ S_new:            # worklist: includes newly added (the ripple)
    for all s ∈ family(o) ∪ N(o):
        if s ∉ S ∪ S_new: S_new ← S_new ∪ {s}
return Linearize(S ∪ S_new)
```

**New preclusion balance (Fig. 7)** — the GPU target:

```
R ← Reduce(S)
R_new ← ∅; R_prec ← ∅
for all o ∈ R:
    for all s ∈ N(o):
        s ← 0-sibling(s)
        if s ∉ R ∪ R_new: R_new ← R_new ∪ {s}
        if s ≺ o:                       R_prec ← R_prec ∪ {s}   # tag precluded
        else if ∃ t ∈ R with t ≺ s:     R_prec ← R_prec ∪ {t}   # tag precluded
R ← R\R_prec; R_new ← R_new\R_prec
return Complete(R ∪ R_new)
```

The win: the costly postprocessing sort runs on a set 2^d smaller; *"requires
roughly 3 times fewer hash queries ... and a reduction of the set that is sorted
in postprocessing by the factor 2^d"*, and constructs **no auxiliary/parent
nodes**.

## Mapping to CUDA / Thrust

Every step is a standard data-parallel primitive — this is why [IBG2012] is the
right target for a GPU port:

| Step | Primitive |
|---|---|
| `Reduce(S)` | `transform` (→ 0-sibling) + `unique`/segmented compaction on sorted codes |
| generate `N(o)` 0-siblings | one kernel, ≤ 2·d outputs/leaf → `copy_if` |
| dedup vs `R ∪ R_new` | `sort` + `unique`, or a device hash set |
| preclusion test `∃ t ∈ R: t ≺ s` | single `lower_bound` (binary search) per candidate |
| `Linearize` / `Complete` | segmented scan over sorted codes (drop ancestors / fill gaps) |

Per-pass work becomes **O(active boundary)**, fixing the wide-front weakness of
active-front ripple. The multi-GPU "one-pass" structure ([IBG2012] §II: local
balance → query → response → rebalance) maps the *Reduce/preclusion* kernel onto
each rank with a **single** insulation-layer exchange round (vs ripple's many).

## Roadmap for this repo

1. **Now:** `balance` (active-front) is production; `balance_ref` is the parity
   oracle. Both face-balance, byte-identical, benchmarked
   (`benchmarks/results/cuda_balance_active_front_*.csv`).
2. **Next:** implement `balance_preclusion` (Fig. 7) in CUDA, **cross-validated
   byte-identical** against `balance_ref` on every fixture before it can become
   the default. Expected to dominate on the wide crack fronts.
3. **Ghosts/multi-GPU:** the recursive ghost construction of [IBWG2015] (works
   without a 2:1 precondition) is the reference for `mpi/` halo tuning.

## Honest gaps (from the verified deep-research pass, 2026-05-29)

- **No public single-GPU CUDA 2:1-balance reference exists.** Every primary
  source ([SSB2008] MPI/Dendro; [BurstWG2011] p4est MPI; csteuer's
  `ParallelBottomUpBalancedOctreeBuilder` OpenMP) is CPU. The *algorithm* is
  fully specified; the GPU port is genuinely ours to make (and is itself a
  publishable contribution).
- The GPU-balance status of Karras-2012 LBVH, AMReX-GPU, GAMER-2, Enzo-E/Cello,
  t8code and dendro-GR **could not be verified** — treat as unknown, not absent.
- All complexity bounds assume near-uniform distributions; our deep-crack load
  imbalance is exactly where those assumptions break, so SFC-partition quality
  (weighted vs equal-count) dominates multi-GPU behaviour and needs measurement.

## Open-source references to read

- **Dendro-5.01** — `github.com/paralab/Dendro-5.01` (Sundar's [SSB2008] impl).
- **p4est** — `github.com/cburstedde/p4est` ([BurstWG2011]; `p?est_balance`).
- **csteuer/ParallelBottomUpBalancedOctreeBuilder** — MIT C++/OpenMP, the most
  readable [SSB2008] port to translate from.
