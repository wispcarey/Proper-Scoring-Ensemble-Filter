#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="/home/bhchen/miniconda3/bin/python"

cd "$REPO_ROOT"

dataset="linear"
seed=42
dim=20
obs_dim=10
test_steps=100

# Toggle figure saving for evaluate_linear.py
# - 0: no figures
# - 1: pass --save_test_figures
save_test_figures=0
save_fig_args=()
if [[ "$save_test_figures" -eq 1 ]]; then
    save_fig_args=(--save_test_figures)
fi

# Ensemble sizes to evaluate
N_list=(5 10 15 20 40 60 100)

# -------------------------
# Trials 1: classic methods (no checkpoint)
# Format: "MethodName SigmaY"
# -------------------------
classic_trials=(
    "LETKF 1"
    "EnKF 1"
)

# -------------------------
# Trials 2: trained methods (require checkpoint)
# Folder names come from save/linear_gaussian_example.
# The lr keys are only for grouping; evaluation uses the folders directly.
# -------------------------
gaussian_example_root="save/linear_gaussian_example"

learned_folders=(
    # "2025-12-25_15-40linear_1_10_60_8192_es_joint_EtE-LRes_20_1e-2"
    # "2025-12-25_15-40linear_1_10_60_8192_l2_joint_EtE-LRes_20_1e-2"
    # "2025-12-25_15-40linear_1_10_60_8192_nl2_joint_EtE-LRes_20_1e-2"
    # "2025-12-25_15-40linear_1_10_60_8192_es_joint_EtE-LRes_20_5e-3"
    # "2025-12-25_15-40linear_1_10_60_8192_l2_joint_EtE-LRes_20_5e-3"
    # "2025-12-25_15-40linear_1_10_60_8192_nl2_joint_EtE-LRes_20_5e-3"
    "2026-01-04_19-49linear_1_10_60_8192_es_joint_EtE-LRes_20_1e-3"
    "2026-01-04_19-49linear_1_10_60_8192_l2_joint_EtE-LRes_20_1e-3"
    "2026-01-04_19-49linear_1_10_60_8192_nl2_joint_EtE-LRes_20_1e-3"
    # "2026-01-04_19-47linear_1_10_60_8192_es_joint_EtE-LRes_20_5e-4"
    # "2026-01-04_19-47linear_1_10_60_8192_l2_joint_EtE-LRes_20_5e-4"
    # "2026-01-04_19-49linear_1_10_60_8192_nes_joint_EtE-LRes_20_5e-4"
)

find_latest_checkpoint() {
    local folder_path="$1"

    "$PYTHON_BIN" -c '
import pathlib
import re
import sys

folder = pathlib.Path(sys.argv[1])
checkpoints = []
for path in folder.glob("cp_*.pth"):
    match = re.fullmatch(r"cp_(\d+)\.pth", path.name)
    if match:
        checkpoints.append((int(match.group(1)), path))

if not checkpoints:
    sys.exit(1)

checkpoints.sort()
print(checkpoints[-1][1].as_posix())
' "$folder_path"
}

echo "=================================================="
echo "Linear eval: classic baselines (no checkpoint)"
echo "Dataset=${dataset}, dim=${dim}, obs_dim=${obs_dim}, test_steps=${test_steps}, seed=${seed}"
echo "=================================================="

for trial in "${classic_trials[@]}"; do
    read -r v sigma_y <<< "$trial"

    echo "--------------------------------------------------"
    echo "Trial type: classic"
    echo "Method: ${v} | sigma_y: ${sigma_y}"
    echo "--------------------------------------------------"

    for N in "${N_list[@]}"; do
        "$PYTHON_BIN" evaluate_linear.py \
            --dataset "$dataset" \
            --N "$N" \
            --sigma_y "$sigma_y" \
            --seed "$seed" \
            --v "$v" \
            --normal_output \
            --test_steps "$test_steps" \
            --dim "$dim" \
            --obs_dim "$obs_dim" \
            "${save_fig_args[@]}"
    done
done

echo "=================================================="
echo "Linear eval: trained methods (with checkpoint)"
echo "=================================================="

for folder_name in "${learned_folders[@]}"; do
    v="EtE-LRes"
    sigma_y=1
    folder_path="${gaussian_example_root}/${folder_name}"

    if [[ ! -d "$folder_path" ]]; then
        echo "[WARN] Folder not found, skip trial: ${folder_path}"
        continue
    fi

    if ! cp_path="$(find_latest_checkpoint "$folder_path")"; then
        echo "[WARN] No checkpoint found, skip trial: ${folder_path}"
        continue
    fi

    echo "--------------------------------------------------"
    echo "Trial type: learned"
    echo "Method: ${v} | sigma_y: ${sigma_y}"
    echo "Folder: ${folder_path}"
    echo "Checkpoint: ${cp_path}"
    echo "--------------------------------------------------"

    for N in "${N_list[@]}"; do
        if [[ "$v" == "EtE-LRes" ]]; then
            "$PYTHON_BIN" evaluate_linear.py \
                --dataset "$dataset" \
                --N "$N" \
                --sigma_y "$sigma_y" \
                --seed "$seed" \
                --v "$v" \
                --no_localization \
                --noise_st_input \
                --normal_output \
                --test_steps "$test_steps" \
                --dim "$dim" \
                --obs_dim "$obs_dim" \
                --cp_load_path "$cp_path" \
                "${save_fig_args[@]}"
        else
            "$PYTHON_BIN" evaluate_linear.py \
                --dataset "$dataset" \
                --N "$N" \
                --sigma_y "$sigma_y" \
                --seed "$seed" \
                --v "$v" \
                --no_localization \
                --normal_output \
                --test_steps "$test_steps" \
                --dim "$dim" \
                --obs_dim "$obs_dim" \
                --cp_load_path "$cp_path" \
                "${save_fig_args[@]}"
        fi
    done
done
