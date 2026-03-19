#!/bin/bash

echo "Building CUDA version..."
make cuda

if [ $? -eq 0 ]; then
    echo "Running CUDA version..."
    ./amr_cuda --cuda
else
    echo "Build failed. Make sure nvcc is in your PATH."
fi
