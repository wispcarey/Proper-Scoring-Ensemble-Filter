#!/bin/bash

cd ..

dataset="lorenz63"
seed=42

# Define experiments as triplets: "MethodName Checkpoint_Path Sigma_Y"
# Format: "MethodName Path/To/Checkpoint SigmaValue"
# experiments=(
#     "CorrTerms save/lorenz63_EtE_LRes/2026-01-14_18-04lorenz63_0.7_10_60_8192_nl2_joint_CorrTerms/cp_1000.pth 0.7"
#     "EtE-LRes save/lorenz63_EtE_LRes/2026-01-14_17-49lorenz63_0.7_10_60_8192_es_joint_EtE-LRes/cp_1000.pth 0.7"
#     "CorrTerms save/lorenz63_EtE_LRes/2026-01-14_18-04lorenz63_0.7_10_60_8192_es_joint_CorrTerms/cp_1000.pth 0.7"
#     "CorrTerms save/lorenz63_EtE_LRes/2026-01-14_01-06lorenz63_1.0_10_60_8192_es_joint_CorrTerms/cp_1000.pth 1.0"
# )

experiments=(
    "EtE-LRes save/lorenz63_wpf/2026-01-21_14-42lorenz63_1.0_10_60_8192_wpf_ed_joint_EtE-LRes/cp_300.pth 1.0"
    "EtE-LRes save/lorenz63_wpf/2026-01-21_14-53lorenz63_1.0_10_60_8192_wpf_ammd_joint_EtE-LRes/cp_300.pth 1.0"
    "EtE-LRes save/lorenz63_wpf/2026-01-21_19-39lorenz63_1.0_10_60_8192_es_wpf_ed_joint_EtE-LRes/cp_300.pth 1.0"
    "EtE-LRes save/lorenz63_wpf/2026-01-21_19-39lorenz63_1.0_10_60_8192_es_wpf_ammd_joint_EtE-LRes/cp_300.pth 1.0"
    "CorrTerms save/lorenz63_wpf/2026-01-21_14-53lorenz63_1.0_10_60_8192_wpf_ed_joint_CorrTerms/cp_300.pth 1.0"
    "CorrTerms save/lorenz63_wpf/2026-01-21_14-53lorenz63_1.0_10_60_8192_wpf_ammd_joint_CorrTerms/cp_300.pth 1.0"
    "CorrTerms save/lorenz63_wpf/2026-01-21_19-39lorenz63_1.0_10_60_8192_es_wpf_ed_joint_CorrTerms/cp_300.pth 1.0"
    "CorrTerms save/lorenz63_wpf/2026-01-21_19-39lorenz63_1.0_10_60_8192_es_wpf_ammd_joint_CorrTerms/cp_300.pth 1.0"
)

# Loop 1: Iterate through configuration
for exp in "${experiments[@]}"; do
    # Extract version, path, and specific sigma_y
    read -r v cp_path current_sigma_y <<< "$exp"
    
    echo "=================================================="
    echo "Evaluating Method: $v"
    echo "Checkpoint: $cp_path"
    echo "Sigma Y: $current_sigma_y"
    echo "=================================================="

    # Loop 2: Iterate through N
    for N in 5 10 15 20 40 60 100; do
        python evaluate.py \
        --dataset $dataset \
        --N $N \
        --sigma_y $current_sigma_y \
        --seed $seed \
        --v $v \
        --no_localization \
        --normal_output \
        --test_steps 500 \
        --pf_verification \
        --pf_N 100000 \
        --sigma_reg None \
        --cp_load_path "$cp_path"
    done
done