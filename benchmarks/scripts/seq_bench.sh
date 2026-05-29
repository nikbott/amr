#!/bin/bash

RESULTS_FILE="thread_benchmark_results_20_12_seq_1.txt"
echo "=== AMR OpenMP Thread Scaling Benchmark ===" > $RESULTS_FILE
echo "Date: $(date)" >> $RESULTS_FILE
echo "-------------------------------------------" >> $RESULTS_FILE

MAX_LEVEL=20
FINE_LEVEL=12

    ./amr_seq --max_level $MAX_LEVEL --fine_level $FINE_LEVEL >> $RESULTS_FILE
    
    echo "-------------------------------------------" >> $RESULTS_FILE

echo "Benchmark complete. Results saved to $RESULTS_FILE"
cat $RESULTS_FILE
