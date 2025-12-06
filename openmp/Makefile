CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -march=native
NVCC = nvcc
NVCCFLAGS = -std=c++17 -O3 --expt-relaxed-constexpr -arch=sm_75

# Detect CUDA
CUDA_AVAILABLE := $(shell which nvcc 2>/dev/null)

SOURCES = main.cpp
HEADERS = config.hpp morton.hpp tree.hpp physics.hpp

ifdef CUDA_AVAILABLE
    TARGET = amr_cuda
    CUDA_SOURCES = tree_kernels.cu
    CUDA_HEADERS = cuda_utils.cuh tree_kernels.cuh
else
    TARGET = amr_cpu
endif

.PHONY: all clean cpu cuda

all: cpu

cpu: $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o amr_cpu main.cpp

cuda: $(SOURCES) $(CUDA_SOURCES) $(HEADERS) $(CUDA_HEADERS)
	$(NVCC) $(NVCCFLAGS) -DUSE_CUDA -o amr_cuda main.cpp $(CUDA_SOURCES)

clean:
	rm -f amr_cpu amr_cuda *.o
