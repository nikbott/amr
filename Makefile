# amr/Makefile — canonical entry points for local development.
# Mirror of what CI will run; designed so wiring CI later is one job per target.
# See docs/CI_FUTURE.md for the GitHub Actions / GitLab CI mapping.

SHELL          := /usr/bin/env bash
BUILD_DIR      ?= build
BUILD_TYPE     ?= Debug
CMAKE_ARGS     ?=
JOBS           ?= $(shell nproc)

# Optional features (default off so a bare `make build` works)
WITH_MPI       ?= ON
WITH_CUDA      ?= OFF
WITH_SANITIZE  ?= ON

CMAKE_FEATURE_FLAGS := \
    -DAMR_BUILD_MPI=$(WITH_MPI) \
    -DAMR_BUILD_CUDA=$(WITH_CUDA) \
    -DENABLE_SANITIZERS=$(WITH_SANITIZE)

.PHONY: help lint format configure build test parity ci-local \
        clean distclean container-cpu container-gpu slurm-strong figs

help: ## list available targets
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

lint: ## run pre-commit hooks on all tracked files
	@if ! command -v pre-commit >/dev/null; then \
	    echo "pre-commit not installed; run: pip install --user pre-commit && pre-commit install"; \
	    exit 1; \
	fi
	pre-commit run --all-files

format: ## reformat C++/CUDA sources via clang-format
	@find omp mpi cuda common 2>/dev/null -type f \
	    \( -name '*.cpp' -o -name '*.hpp' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' \) \
	    -print0 | xargs -0 -r clang-format -i

configure: ## cmake configure (re-runs if BUILD_TYPE changes)
	cmake -B $(BUILD_DIR) -S . -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) $(CMAKE_FEATURE_FLAGS) $(CMAKE_ARGS)

build: configure ## build all enabled backends
	cmake --build $(BUILD_DIR) -j $(JOBS)

test: build ## run C++/CUDA unit tests via ctest
	ctest --test-dir $(BUILD_DIR) --output-on-failure

parity: build ## cross-backend parity test (OMP ↔ MPI ↔ CUDA)
	@if [ -x scripts/parity.py ]; then \
	    python3 scripts/parity.py --backends omp,mpi; \
	else \
	    echo "[stub] parity test added in Stage 2 — skipping"; \
	fi

ci-local: lint test parity ## full local verification (matches CI)
	@echo "ci-local: green"

# --- Containers ---------------------------------------------------------------
container-cpu: ## build amr-cpu Apptainer image (Stage 2)
	@if [ -f container/amr-cpu.def ]; then \
	    apptainer build container/amr-cpu_$$(git rev-parse --short HEAD).sif container/amr-cpu.def; \
	else \
	    echo "[stub] container/amr-cpu.def added in Stage 2"; \
	fi

container-gpu: ## build amr-gpu Apptainer image (Stage 2)
	@if [ -f container/amr-gpu.def ]; then \
	    apptainer build container/amr-gpu_$$(git rev-parse --short HEAD).sif container/amr-gpu.def; \
	else \
	    echo "[stub] container/amr-gpu.def added in Stage 2"; \
	fi

# --- SLURM (Stage 2) ----------------------------------------------------------
slurm-strong: ## submit strong-scaling benchmark to SLURM
	@if [ -f slurm/02_strong_scaling.sbatch ]; then \
	    sbatch slurm/02_strong_scaling.sbatch; \
	else \
	    echo "[stub] slurm/02_strong_scaling.sbatch added in Stage 2"; \
	fi

figs: ## regenerate paper figures from results/
	@if [ -x slurm/postprocess.py ]; then \
	    python slurm/postprocess.py --in results/ --out docs/figures/; \
	else \
	    echo "[stub] slurm/postprocess.py added in Stage 2"; \
	fi

# --- Cleanup ------------------------------------------------------------------
clean: ## remove build/ but keep configure cache
	rm -rf $(BUILD_DIR)/CMakeFiles $(BUILD_DIR)/Testing
	@find $(BUILD_DIR) -maxdepth 2 -type f \( -name '*.o' -o -name '*.a' \) -delete 2>/dev/null || true

distclean: ## nuke build/ entirely
	rm -rf $(BUILD_DIR)
