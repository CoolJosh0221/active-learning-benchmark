#!/bin/bash

# Python interpreter path
PYTHON_EXEC="/home/hosh/mambaforge/envs/active-learning-benchmark/bin/python"

# Create results directory if it doesn't exist
mkdir -p ../results

echo "Starting experiments sequentially in background..."

# Run all top 3 datasets one by one in a single background process
# This prevents GPU OOM by ensuring only one TabPFN model is loaded at a time
nohup bash -c "
    echo 'Starting Splice...' && \
    $PYTHON_EXEC -u run_tabpfn_experiments.py --datasets splice > ../results/splice_experiment_full.log 2>&1 && \
    echo 'Starting Ionosphere...' && \
    $PYTHON_EXEC -u run_tabpfn_experiments.py --datasets ionosphere > ../results/ionosphere_experiment_full.log 2>&1 && \
    echo 'Starting Pol...' && \
    $PYTHON_EXEC -u run_tabpfn_experiments.py --datasets pol > ../results/pol_experiment_full.log 2>&1
" > ../results/tabpfn_sequential_master.log 2>&1 &

echo "Sequential master process launched with PID: $!"
echo "Check ../results/tabpfn_sequential_master.log for overall status."
echo "Check ../results/[dataset]_experiment_full.log for detailed progress."