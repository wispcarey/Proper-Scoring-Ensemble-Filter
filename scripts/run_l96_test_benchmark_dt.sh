#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/bhchen/miniconda3/bin/python}"

dataset="lorenz96"
sigma_y="1.0"
seed="${SEED:-42}"
N=10
device="${DEVICE:-cpu}"
num_loader_workers="${NUM_LOADER_WORKERS:-8}"

obs_fns=(identity arctan square)
methods=(EnKF ESRF LETKF iEnKS-PertObs iEnKS-Sqrt)
dt_settings=(
    "0.15 5"
    "0.18 6"
    "0.21 7"
    "0.24 8"
    "0.27 9"
    "0.30 10"
    "0.33 11"
    "0.36 12"
    "0.39 13"
    "0.42 14"
    "0.45 15"
)

echo "Date: $(date)"
echo "Python: $PYTHON_BIN"
echo "Dataset: $dataset"
echo "Sigma Y: $sigma_y"
echo "N: $N"
echo "Device: $device"
echo "Obs Fns: ${obs_fns[*]}"
echo "Methods: ${methods[*]}"
echo "dt settings: ${dt_settings[*]}"
echo "----------------------------------------------------"

for dt_setting in "${dt_settings[@]}"; do
    read -r dt dt_iter <<< "$dt_setting"
    output_suffix="_dt${dt}"

    for obs_fn in "${obs_fns[@]}"; do
        for method in "${methods[@]}"; do
            echo "============================================================"
            echo "Running benchmark test: dataset=$dataset obs_fn=$obs_fn N=$N method=$method dt=$dt dt_iter=$dt_iter"
            echo "Output suffix: $output_suffix"
            echo "============================================================"

            cmd=(
                "$PYTHON_BIN" evaluate_benchmark.py
                --device "$device"
                --dataset "$dataset"
                --N "$N"
                --sigma_y "$sigma_y"
                --seed "$seed"
                --v "$method"
                --obs_fn "$obs_fn"
                --dt "$dt"
                --dt_iter "$dt_iter"
                --num_loader_workers "$num_loader_workers"
                --normal_output
                --save_test_figures
                --adaptive_sigma_y
                --suffix "$output_suffix"
            )

            "${cmd[@]}"

            echo "Finished obs_fn=$obs_fn N=$N method=$method dt=$dt."
            echo ""
        done
    done
done

echo "All Lorenz96 benchmark dt tests finished."
echo "Completed on $(date)"
