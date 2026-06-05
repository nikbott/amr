# Contributing to AMR

Thanks for your interest in the AMR engine. This is research software that
doubles as the mesh backend for an adaptive DIC/DVC solver, so correctness and
reproducibility matter more than feature count.

## Ground rules

- **One logical change per branch and per pull request.** No mega-PRs.
- **Branch naming:** `<type>/<slug>` — e.g. `feat/scalar-field-oracle`,
  `perf/mpi-neighbor-ghost`, `fix/shift-by-64`, `docs/balance-notes`,
  `chore/...`, `ci/...`.
- **Commits** follow [Conventional Commits](https://www.conventionalcommits.org/)
  with short bodies explaining *why*. Sign your commits and tags
  (`git config commit.gpgsign true`).
- **`dev` is the trunk.** Open PRs against `dev`; never push directly to it.
  `main` is reconciled from `dev`.

## Development setup

```bash
pip install --user pre-commit && pre-commit install   # hooks: format + lint
make build         # cmake configure + build enabled backends
make test          # ctest (Catch2 / CUDA harness)
make parity        # cross-backend parity (omp ↔ mpi ↔ cuda)
make ci-local      # lint + test + parity (mirrors CI)
```

Backends and how to build each are in [README.md](README.md). The Morton core,
the sorted-leaf invariants, and how to add an oracle are documented there and in
`docs/`.

## What CI checks

- **lint** — pre-commit (clang-format, cmake-format, codespell, hygiene) on the
  files your PR changes. Run `pre-commit run --from-ref origin/dev --to-ref HEAD`
  locally to preview.
- **cpu** — gcc/clang × Debug/Release, OpenMP + MPI, ctest, sanitizers on Debug.
- **cuda-build** — compile-only (no GPU on hosted runners).
- **gpu** — optional, runs only when a self-hosted GPU runner is online.
- **codeql** — static security analysis (C/C++).

A PR must keep CI green. The lint job only inspects changed files, so you will
not be asked to reformat code you did not touch.

## Tests are not optional

Every behavioural change lands with a test. The C++ backends cross-check each
other and the in-tree `balance_ref()` brute-force oracle; keep that net intact.

## Reporting bugs / proposing work

Use the issue templates (bug report, feature request, research task). For
anything security-sensitive, see [SECURITY.md](SECURITY.md).
