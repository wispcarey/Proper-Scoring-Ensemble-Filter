#!/bin/bash

# Script to run finetuning with variable loss types.
# Input: Quadruplets "Method Checkpoint Sigma LossType"
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

# Define experiments as triplets: "MethodName Checkpoint_Path Sigma_Y"
experiments=(
    "CorrTerms save/lorenz96_EtE_LRes/2026-01-14_01-24lorenz96_1.0_10_60_8192_es_joint_CorrTerms/cp_1000.pth 1.0 es"
    "EtE-LRes save/lorenz96_EtE_LRes/2026-01-07_16-42lorenz96_1.0_10_60_8192_es_joint_EtE-LRes/cp_1000.pth 1.0 es"
    "CorrTerms save/lorenz96_EtE_LRes/2026-01-07_17-39lorenz96_1.0_10_60_8192_nl2_joint_CorrTerms/cp_1000.pth 1.0 nl2"
)

# Loop: Iterate through configuration
for exp in "${experiments[@]}"; do
    # Extract params including loss_type
    read -r v cp_path current_sigma_y loss_type <<< "$exp"
    
    echo "=================================================="
    echo "Finetuning Method: $v"
    echo "Loss Type: $loss_type (es_p=$es_p)"
    echo "Sigma Y: $current_sigma_y"
    echo "=================================================="

    python finetune.py \
        --epochs $epochs \
        --save_epoch $save_epoch \
        --dataset $dataset \
        --train_steps $train_steps \
        --train_traj_num $train_traj_num \
        --sigma_y $current_sigma_y \
        --seed $seed \
        --learning_rate $learning_rate \
        --cp_load_path "$cp_path" \
        --v $v \
        --normal_output \
        --loss_type $loss_type \
        --es_p $es_p \
        --no_running_loss
done