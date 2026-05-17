#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/bhchen/miniconda3/bin/python}"

dataset="lorenz63"
# Keep this aligned with DATASET_INFO['lorenz63']['sigma_y'] when using
# --adaptive_sigma_y. The final sigma_y is adapted per obs_fn.
sigma_y="2.0"
seed="${SEED:-42}"
N=10
device="${DEVICE:-cuda}"
num_loader_workers="${NUM_LOADER_WORKERS:-8}"
pf_N="${PF_N:-1000000}"
save_test_figures="${SAVE_TEST_FIGURES:-0}"

plot_start_step="${PLOT_START_STEP:-1}"
plot_interval_step="${PLOT_INTERVAL_STEP:-1}"
plot_end_step="${PLOT_END_STEP:-500}"

obs_fns=(identity arctan square)
methods=(EnKF ESRF iEnKS-PertObs iEnKS-Sqrt iEnKS-Order1)
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
echo "Base Sigma Y: $sigma_y"
echo "N: $N"
echo "Device: $device"
echo "PF_N: $pf_N"
echo "Save Test Figures: $save_test_figures"
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
            echo "Running L63 benchmark PF dt test: dataset=$dataset obs_fn=$obs_fn N=$N method=$method dt=$dt dt_iter=$dt_iter"
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
                --no_localization
                --normal_output
                --pf_verification
                --pf_N "$pf_N"
                --adaptive_sigma_y
                --suffix "$output_suffix"
            )

            if [[ "$save_test_figures" != "0" && "$save_test_figures" != "false" && "$save_test_figures" != "False" ]]; then
                cmd+=(
                    --save_test_figures
                    --test_snapshot_start_step "$plot_start_step"
                    --test_snapshot_interval "$plot_interval_step"
                    --test_snapshot_end_step "$plot_end_step"
                )
            fi

            "${cmd[@]}"

            echo "Finished obs_fn=$obs_fn N=$N method=$method dt=$dt."
            echo ""
        done
    done
done

echo "All Lorenz63 benchmark PF dt tests finished."
echo "Completed on $(date)"
