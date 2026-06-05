<!-- One logical change per PR. See CONTRIBUTING.md. -->

## What & why

<!-- What does this change do, and why? Link the issue it closes: Closes #NNN -->

## Type

- [ ] feat — new capability
- [ ] fix — bug fix
- [ ] perf — performance (include before/after numbers)
- [ ] refactor / chore / docs / ci

## Checklist

- [ ] Branch is `<type>/<slug>`; commits are Conventional + signed
- [ ] One logical change (no unrelated edits)
- [ ] Tests added/updated; `make ci-local` passes locally
- [ ] Backends kept in parity (omp ↔ mpi ↔ cuda) where applicable
- [ ] Docs / README updated if behaviour or layout changed

## Benchmarks (perf PRs)

<!-- backend, host, problem size, before → after, speedup -->
