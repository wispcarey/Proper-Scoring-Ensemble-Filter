#!/bin/bash

set -euo pipefail

cd ..

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
N_list=(10)

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
# Format: "MethodName CheckpointPath SigmaY"
# -------------------------
learned_trials=(
    "CorrTerms save/linear_es_vs_l2/2025-08-14_20-05linear_1_10_60_8192_es_joint_CorrTerms/cp_1000.pth 1"
    "CorrTerms save/linear_es_vs_l2/2025-08-14_20-04linear_1_10_60_8192_l2_joint_CorrTerms/cp_1000.pth 1"
    "EtE-LRes save/linear_es_vs_l2/2025-08-14_20-04linear_1_10_60_8192_l2_joint_EtE-LRes_nst/cp_1000.pth 1"
    "EtE-LRes save/linear_es_vs_l2/2025-08-14_19-59linear_1_10_60_8192_es_joint_EtE-LRes_nst/cp_1000.pth 1"
)

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
        python evaluate_linear.py \
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

for trial in "${learned_trials[@]}"; do
    read -r v cp_path sigma_y <<< "$trial"

    if [[ ! -f "$cp_path" ]]; then
        echo "[WARN] Checkpoint not found, skip trial: ${cp_path}"
        continue
    fi

    echo "--------------------------------------------------"
    echo "Trial type: learned"
    echo "Method: ${v} | sigma_y: ${sigma_y}"
    echo "Checkpoint: ${cp_path}"
    echo "--------------------------------------------------"

    for N in "${N_list[@]}"; do
        if [[ "$v" == "EtE-LRes" ]]; then
            python evaluate_linear.py \
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
            python evaluate_linear.py \
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
