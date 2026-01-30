#!/bin/bash

cd ..

dataset="lorenz96"
seed=42

# Define experiments as triplets: "MethodName Checkpoint_Directory Sigma_Y"
# Note: The path is now the directory containing the checkpoints, not the file itself
experiments=(
    "EtE-LRes save/lorenz96_EtE_LRes/2026-01-17_13-10lorenz96_1.0_20_60_8192_es_joint_EtE-LRes_tuned 1.0"
    "CorrTerms save/lorenz96_EtE_LRes/2026-01-16_13-33lorenz96_1.0_20_60_8192_es_joint_CorrTerms_tuned 1.0"
    "CorrTerms save/lorenz96_EtE_LRes/2026-01-17_01-16lorenz96_1.0_20_60_8192_nl2_joint_CorrTerms_tuned 1.0"
)

# Loop 1: Iterate through configuration
for exp in "${experiments[@]}"; do
    # Extract version, base directory, and specific sigma_y
    read -r v base_dir current_sigma_y <<< "$exp"
    
    echo "=================================================="
    echo "Evaluating Method: $v"
    echo "Base Directory: $base_dir"
    echo "Sigma Y: $current_sigma_y"
    echo "=================================================="

    # Loop 2: Iterate through N
    for N in 5 10 15 20 40 60 100; do
        # Construct dynamic checkpoint path based on N
        # Format: ft_cp_{N}_20.pth
        cp_path="${base_dir}/ft_cp_${N}_20.pth"

        python evaluate.py \
        --dataset $dataset \
        --N $N \
        --sigma_y $current_sigma_y \
        --seed $seed \
        --v $v \
        --normal_output \
        --test_steps 500 \
        --sigma_reg None \
        --cp_load_path "$cp_path" \
        --sigma_ens 5 \
        --suffix "sigma_ens_5"
    done
done