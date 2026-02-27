#!/bin/bash

set -e

cd ..

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    else
        echo "Error: neither 'python' nor 'python3' is available in PATH." >&2
        exit 127
    fi
fi

echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Python: $(command -v "$PYTHON_BIN")"
echo "----------------------------------------------------"

# Format:
#   "dataset obs_fn sigma_mode test_steps test_traj_num"
# sigma_mode:
#   - "adaptive"          => pass --adaptive_sigma_y
#   - numeric (e.g. 1.0)  => pass --sigma_y <value>
experiments=(
    "lorenz63 square adaptive 500 64"
    # "lorenz63 arctan adaptive 500 64"
    # "lorenz63 default adaptive 500 64"
)

for exp in "${experiments[@]}"; do
    read -r dataset obs_fn sigma_mode test_steps test_traj_num <<< "$exp"

    echo "============================================================"
    echo "Running PF analysis"
    echo "Dataset: $dataset"
    echo "Obs Fn: $obs_fn"
    echo "Sigma Mode: $sigma_mode"
    echo "Test Steps: $test_steps"
    echo "Test Traj Num: $test_traj_num"
    echo "============================================================"

    cmd=(
        "$PYTHON_BIN" analyze_pf_results.py
        --device "cpu"
        --dataset "$dataset"
        --obs_fn "$obs_fn"
        --test_steps "$test_steps"
        --test_traj_num "$test_traj_num"
        --normal_output
    )

    if [ "$sigma_mode" = "adaptive" ]; then
        cmd+=(--adaptive_sigma_y)
    else
        cmd+=(--sigma_y "$sigma_mode")
    fi

    "${cmd[@]}"
    echo "Done."
    echo ""
done

echo "All PF analysis runs finished."
echo "Job completed on $(date)"
