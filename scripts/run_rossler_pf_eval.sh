#!/bin/bash

cd ..

# Rossler
dataset="rossler"

sigma_y=1
seed=42

# EtE-LRes es1 nst
for N in 5 10 15 20 40 60 100; do
    python evaluate.py \
    --dataset $dataset \
    --N $N \
    --sigma_y $sigma_y \
    --seed $seed \
    --v EtE-LRes \
    --no_localization \
    --normal_output \
    --test_steps 500 \
    --pf_verification \
    --pf_N 100000 \
    --sigma_reg None \
    --cp_load_path save/rossler_EtE_LRes/2025-08-08_09-55rossler_1.0_10_60_8192_es_joint_EtE-LRes_nst/cp_1000.pth \
    --noise_st_input 
done

sigma_y=1
seed=42
save_dir="rossler_EtE_LRes/2025-08-14_20-29rossler_1.0_20_60_8192_nl2_joint_EtE-LRes_nst_tuned"

# sigma_y = $sigma_y, EnST
for N in 5 10 15 20 40 60 100; do
    python evaluate.py \
        --dataset $dataset \
        --N $N \
        --v EtE-LRes \
        --sigma_y $sigma_y \
        --seed $seed \
        --cp_load_path save/${save_dir}/ft_cp_${N}_20.pth \
        --no_localization \
        --pf_verification \
        --pf_N 100000 \
        --sigma_reg None \
        --noise_st_input
done