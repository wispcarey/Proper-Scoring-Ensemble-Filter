#!/bin/bash

cd ..

# lorenz 63
dataset="lorenz63"

sigma_y=1
seed=42

# corrterms es1
for N in 5 10 15 20 40 60 100; do
    python evaluate.py \
    --dataset $dataset \
    --N $N \
    --sigma_y $sigma_y \
    --seed $seed \
    --v CorrTerms \
    --no_localization \
    --normal_output \
    --test_steps 500 \
    --pf_verification \
    --pf_N 100000 \
    --sigma_reg None \
    --cp_load_path save/2025-06-28_21-55lorenz63_1.0_10_60_8192_es_joint_CorrTerms/cp_1000.pth
done

# corrterms l2
for N in 5 10 15 20 40 60 100; do
    python evaluate.py \
    --dataset $dataset \
    --N $N \
    --sigma_y $sigma_y \
    --seed $seed \
    --v CorrTerms \
    --no_localization \
    --normal_output \
    --test_steps 500 \
    --pf_verification \
    --pf_N 100000 \
    --sigma_reg None \
    --cp_load_path save/2025-06-28_21-52lorenz63_1.0_10_60_8192_l2_joint_CorrTerms/cp_1000.pth
done

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
    --cp_load_path save/lorenz63_EtE_LRes/2025-07-09_19-08lorenz63_1.0_10_60_8192_es_joint_EtE-LRes_nst/cp_1000.pth \
    --noise_st_input 
done

sigma_y=1
seed=42
save_dir="lorenz63_EtE_LRes/2025-08-08_09-55lorenz63_1.0_20_60_8192_nl2_joint_EtE-LRes_nst_tuned"

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