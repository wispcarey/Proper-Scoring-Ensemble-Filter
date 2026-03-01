#!/bin/bash

# Script to run finetuning with variable loss types.
# Input: Sextuplets "Method Checkpoint Sigma LossType ObsFn AdaptiveSigmaY"
# Output: Runs finetune.py with specific loss configuration.

cd ..

dataset="lorenz96"
seed=42
epochs=20
save_epoch=20
train_steps=60
train_traj_num=8192
learning_rate="1e-4"
es_p=1

# Define experiments as sextuplets:
# "MethodName Checkpoint_Path Sigma_Y LossType ObsFn AdaptiveSigmaY"
experiments=(
    "EtE-LRes save/l96_varying_obs_fn/2026-02-22_20-32lorenz96_6.69_10_60_8192_es_joint_EtE-LResNone_square/cp_1000.pth default es square true"
    "CorrTerms save/l96_varying_obs_fn/2026-02-28_14-42lorenz96_1.0_10_60_8192_nl2_joint_CorrTermsNone_identity/cp_1000.pth default nl2 default true"
)

validate_adaptive_sigma_y() {
    case "${1,,}" in
        true|false|1|0|yes|no|y|n) return 0 ;;
        *)
            echo "Error: ADAPTIVE_SIGMA_Y must be a boolean-like value. Got '$1'" >&2
            return 1
            ;;
    esac
}

# Loop: Iterate through configuration
for exp in "${experiments[@]}"; do
    # Extract params including obs_fn and adaptive_sigma_y
    read -r v cp_path current_sigma_y loss_type current_obs_fn adaptive_sigma_y <<< "$exp"

    current_obs_fn=${current_obs_fn:-default}
    adaptive_sigma_y=${adaptive_sigma_y:-true}

    if ! validate_adaptive_sigma_y "$adaptive_sigma_y"; then
        echo "Malformed experiment entry: $exp" >&2
        exit 1
    fi
    
    echo "=================================================="
    echo "Finetuning Method: $v"
    echo "Loss Type: $loss_type (es_p=$es_p)"
    echo "Sigma Y: $current_sigma_y"
    echo "Obs Fn: $current_obs_fn"
    echo "Adaptive Sigma Y: $adaptive_sigma_y"
    echo "=================================================="

    cmd=(
        python finetune.py
        --epochs "$epochs"
        --save_epoch "$save_epoch"
        --dataset "$dataset"
        --train_steps "$train_steps"
        --train_traj_num "$train_traj_num"
        --sigma_y "$current_sigma_y"
        --seed "$seed"
        --learning_rate "$learning_rate"
        --cp_load_path "$cp_path"
        --v "$v"
        --normal_output
        --obs_fn "$current_obs_fn"
        --loss_type "$loss_type"
        --es_p "$es_p"
        --no_running_loss
    )

    case "${adaptive_sigma_y,,}" in
        true|1|yes|y)
            cmd+=(--adaptive_sigma_y)
            ;;
    esac

    "${cmd[@]}"
done
