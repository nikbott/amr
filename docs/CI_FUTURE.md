# CI_FUTURE.md — drop-in CI workflows for `amr/`

Local development is driven by `make` targets (`make ci-local`, `make figs`,
`make container-gpu`, …). This file maps those targets to GitHub Actions and
GitLab CI YAML, so wiring CI on later is a copy-paste, not a redesign.

Until CI is on:
- `pre-commit install` runs the hooks on every commit (fast safety net).
- `make ci-local` is the manual full check before pushing.

## GitHub Actions

Drop this at `.github/workflows/ci.yml`:

```yaml
name: amr CI
on:
  push:
    branches: [main, cpp-sequential]
  pull_request:

jobs:
  cpu:
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-22.04, ubuntu-24.04]
        cxx: [g++-13, clang++-17]
        build: [Debug, Release]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/cache@v4
        with:
          path: ~/.ccache
          key: ccache-${{ matrix.os }}-${{ matrix.cxx }}-${{ matrix.build }}
      - name: Install deps
        run: |
          sudo apt-get update
          sudo apt-get install -y cmake ninja-build openmpi-bin libopenmpi-dev \
                                  libomp-dev ccache python3-pip pre-commit
          pip install -r python/requirements.txt
      - name: Lint
        run: pre-commit run --all-files
      - name: Build + test (CPU)
        env:
          CXX: ${{ matrix.cxx }}
        run: make ci-local BUILD_TYPE=${{ matrix.build }} WITH_CUDA=OFF
      - name: Upload ctest logs on failure
        if: failure()
        uses: actions/upload-artifact@v4
        with:
          name: ctest-log-${{ matrix.os }}-${{ matrix.cxx }}-${{ matrix.build }}
          path: build/Testing/Temporary/LastTest.log

  gpu:
    runs-on: [self-hosted, linux, gpu]
    if: github.event_name == 'push' || github.event_name == 'schedule'
    steps:
      - uses: actions/checkout@v4
      - name: Build + test (CUDA)
        run: make ci-local WITH_CUDA=ON BUILD_TYPE=Release
      - name: Profile
        run: |
          nsys profile -o build/nsys --stats=true build/amr_bench || true
          ncu --set roofline build/amr_bench --kernel-name regex:k_balance || true
      - uses: actions/upload-artifact@v4
        with:
          name: gpu-profiles
          path: build/{nsys.qdrep,ncu.ncu-rep}

  release:
    needs: [cpu, gpu]
    if: startsWith(github.ref, 'refs/tags/v')
    runs-on: [self-hosted, linux, gpu]
    steps:
      - uses: actions/checkout@v4
      - run: make container-cpu container-gpu
      - run: make figs
      - uses: softprops/action-gh-release@v2
        with:
          files: |
            container/*.sif
            docs/figures/*.pdf
```

Notes:
- The `gpu` job needs a self-hosted runner labeled `gpu` with CUDA + an NVIDIA
  device. Set it up on a lab machine; document the runner registration command
  in this file when the runner exists.
- `make ci-local` already encapsulates lint → build → unit tests → Python tests
  → parity test. CI should not duplicate steps.
- The `release` job is tag-triggered (`refs/tags/v*`). Use semantic versions
  (`v1.0.0`); Apptainer images carry the git SHA in their filename.

## GitLab CI (lab-hosted, e.g. Centrale-Supélec or UFSCar GitLab)

Drop this at `.gitlab-ci.yml`:

```yaml
stages: [lint, build, test, container, release]

variables:
  CCACHE_DIR: $CI_PROJECT_DIR/.ccache

cache:
  key: ccache
  paths: [.ccache/]

lint:
  stage: lint
  image: python:3.12-slim
  before_script: [pip install pre-commit]
  script: [pre-commit run --all-files]

.cpu_matrix:
  stage: build
  parallel:
    matrix:
      - { CXX: g++,    BUILD_TYPE: Debug }
      - { CXX: g++,    BUILD_TYPE: Release }
      - { CXX: clang++, BUILD_TYPE: Debug }

build_cpu:
  extends: .cpu_matrix
  image: ubuntu:24.04
  before_script:
    - apt-get update && apt-get install -y cmake make $CXX openmpi-bin libopenmpi-dev libomp-dev python3-pip ccache
    - pip install -r python/requirements.txt
  script: [make ci-local BUILD_TYPE=$BUILD_TYPE WITH_CUDA=OFF]

build_gpu:
  stage: build
  tags: [cuda]                           # GitLab runner with --tag cuda
  image: nvidia/cuda:12.4.0-devel-ubuntu22.04
  before_script:
    - apt-get update && apt-get install -y cmake openmpi-bin libopenmpi-dev libomp-dev
  script: [make ci-local WITH_CUDA=ON BUILD_TYPE=Release]
  artifacts:
    paths: [build/Testing/Temporary/LastTest.log]
    when: on_failure

apptainer:
  stage: container
  tags: [apptainer]
  rules: [{ if: $CI_COMMIT_TAG }]
  script:
    - make container-cpu container-gpu
  artifacts: { paths: [container/*.sif], expire_in: never }

release:
  stage: release
  rules: [{ if: $CI_COMMIT_TAG }]
  script: [make figs]
  artifacts: { paths: [docs/figures/*.pdf] }
```

## Wiring later (checklist)

1. Decide platform (GitHub Actions vs GitLab CI) — see `docs/CI_FUTURE.md` in the
   repo root: it tracks the decision once made.
2. Register a self-hosted runner with CUDA + Apptainer for the GPU jobs.
3. Copy the relevant YAML into the right location.
4. Push to a feature branch and verify each `make` target succeeds in CI before
   merging.
5. Add a `CI` badge to `README.md`.

Everything in this file is reversible — deleting `docs/CI_FUTURE.md` and the
workflow files takes the repo back to local-only.
