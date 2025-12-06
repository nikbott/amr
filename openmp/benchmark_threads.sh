#!/bin/bash

RESULTS_FILE="thread_benchmark_results.txt"
echo "=== AMR OpenMP Thread Scaling Benchmark ===" > $RESULTS_FILE
echo "Date: $(date)" >> $RESULTS_FILE
echo "-------------------------------------------" >> $RESULTS_FILE

# Using Level 15 as it provides a good balance of duration (approx 1.5s sequential)
MAX_LEVEL=20
FINE_LEVEL=12

for threads in 1 2 4 8 16 32 64; do
    echo "Running with OMP_NUM_THREADS=$threads..."
    echo "" >> $RESULTS_FILE
    echo "### Threads: $threads" >> $RESULTS_FILE
    
    export OMP_NUM_THREADS=$threads
    ./build/amr --max_level $MAX_LEVEL --fine_level $FINE_LEVEL >> $RESULTS_FILE
    
    echo "-------------------------------------------" >> $RESULTS_FILE
done

echo "Benchmark complete. Results saved to $RESULTS_FILE"
cat $RESULTS_FILE
