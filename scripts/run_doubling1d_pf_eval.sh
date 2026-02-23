#!/bin/bash

cd ..

dataset="doubling1d"
seed=42

# Define experiments as triplets: "MethodName Checkpoint_Path Sigma_Y"
# Format: "MethodName Path/To/Checkpoint SigmaValue"

experiments=(
    "CorrTerms save/doubling1d_results/2026-02-12_19-24doubling1d_0.2_30_60_8192_es_joint_CorrTermsNone_cos2pi/cp_500.pth 0.2"
    "EtE-LRes save/doubling1d_results/2026-02-12_19-24doubling1d_0.2_30_60_8192_es_joint_EtE-LResNone_cos2pi/cp_500.pth 0.2"
    "CorrTerms save/doubling1d_results/2026-02-12_19-24doubling1d_0.2_30_60_8192_nl2_joint_CorrTermsNone_cos2pi/cp_500.pth 0.2"
    "EtE-LRes save/doubling1d_results/2026-02-12_19-24doubling1d_0.2_30_60_8192_nl2_joint_EtE-LResNone_cos2pi/cp_500.pth 0.2"
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
    for N in 10 30 100; do
        python evaluate.py \
        --dataset $dataset \
        --N $N \
        --sigma_y $current_sigma_y \
        --seed $seed \
        --v $v \
        --no_localization \
        --normal_output \
        --pf_verification \
        --pf_N 1000000 \
        --sigma_reg None \
        --cp_load_path "$cp_path" \
        --save_test_figures 
    done
done