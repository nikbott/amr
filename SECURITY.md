# Security Policy

AMR is academic research software, but we take defects that could compromise a
host (memory-safety bugs in the C++/CUDA backends, unsafe handling of untrusted
mesh input, CI/supply-chain issues) seriously.

## Reporting a vulnerability

Please **do not** open a public issue for a security problem.

- Preferred: open a [private security advisory](https://github.com/nikbott/amr/security/advisories/new)
  via GitHub.
- Or email the maintainer at **the maintainer (via GitHub)** with a description,
  reproduction steps, and the affected commit.

We aim to acknowledge within a week. Because this is a small research project,
fixes are best-effort and tracked in the open once a patch exists.

## Scope

- In scope: the `omp/`, `mpi/`, and `cuda/` backends, the binary mesh I/O, the
  build system, and the CI workflows.
- Out of scope: third-party dependencies (report upstream), and the separate
  MATLAB DIC repository that consumes this engine.

## Supported versions

Only the tip of `dev` (and the latest tagged release) receives fixes.
