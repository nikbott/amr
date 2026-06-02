# CUDA backend build.
#
# NOTE: the Thrust transform/scan calls in tree.cuh use __device__ lambdas,
# which require nvcc's --extended-lambda flag. Without it the build fails with
#   "__host__ or __device__ annotation on lambda requires --extended-lambda".
#
# Override ARCH for your GPU, e.g. `make ARCH=sm_86` (RTX A2000/Ampere),
# `sm_80` (A100), `sm_90` (H100).

NVCC      ?= nvcc
ARCH      ?= sm_70
NVCCFLAGS  = -std=c++17 --extended-lambda -O3 -arch=$(ARCH)

TARGET   = amr
TESTS    = test
BENCH    = bench
HEADERS  = core.cuh tree.cuh physics.cuh viz.cuh

all: $(TARGET) $(TESTS) $(BENCH)

$(TARGET): main.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) main.cu -o $(TARGET)

$(TESTS): tests.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) tests.cu -o $(TESTS)

# Active-front vs reference balance() benchmark (cuda/bench.cu).
$(BENCH): bench.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) bench.cu -o $(BENCH)

clean:
	rm -f $(TARGET) $(TESTS) $(BENCH) *.svg
