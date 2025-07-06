#!/bin/bash

cd ..

# lorenz 63
dataset="lorenz63"

sigma_y=1
seed=42

# corrterms es1
save_dir="2025-06-28_21-55lorenz63_1.0_10_60_8192_es_joint_CorrTerms"
for N in 10; do
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
    --pf_N 1000000 \
    --sigma_reg None \
    --cp_load_path save/${save_dir}/cp_1000.pth
done

# corrterms es2
save_dir="2025-06-28_21-52lorenz63_1.0_10_60_8192_l2_joint_CorrTerms"
for N in 10; do
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
    --pf_N 1000000 \
    --sigma_reg None \
    --cp_load_path save/${save_dir}/cp_1000.pth
done

# ete es1
save_dir="2025-06-28_21-55lorenz63_1.0_10_60_8192_es_joint_EtE"
for N in 10; do
    python evaluate.py \
    --dataset $dataset \
    --N $N \
    --sigma_y $sigma_y \
    --seed $seed \
    --v EtE \
    --no_localization \
    --normal_output \
    --test_steps 500 \
    --pf_verification \
    --pf_N 1000000 \
    --sigma_reg None \
    --cp_load_path save/${save_dir}/cp_1000.pth
done

# ete es2
save_dir="2025-06-28_21-52lorenz63_1.0_10_60_8192_l2_joint_EtE"
for N in 10; do
    python evaluate.py \
    --dataset $dataset \
    --N $N \
    --sigma_y $sigma_y \
    --seed $seed \
    --v EtE \
    --no_localization \
    --normal_output \
    --test_steps 500 \
    --pf_verification \
    --pf_N 1000000 \
    --sigma_reg None \
    --cp_load_path save/${save_dir}/cp_1000.pth
done

# ete es1
save_dir="2025-06-28_21-55lorenz63_1.0_10_60_8192_es_joint_EtE2"
for N in 10; do
    python evaluate.py \
    --dataset $dataset \
    --N $N \
    --sigma_y $sigma_y \
    --seed $seed \
    --v EtE2 \
    --no_localization \
    --normal_output \
    --test_steps 500 \
    --pf_verification \
    --pf_N 1000000 \
    --sigma_reg None \
    --cp_load_path save/${save_dir}/cp_1000.pth
done

# ete es2
save_dir="2025-06-28_21-52lorenz63_1.0_10_60_8192_l2_joint_EtE2"
for N in 10; do
    python evaluate.py \
    --dataset $dataset \
    --N $N \
    --sigma_y $sigma_y \
    --seed $seed \
    --v EtE2 \
    --no_localization \
    --normal_output \
    --test_steps 500 \
    --pf_verification \
    --pf_N 1000000 \
    --sigma_reg None \
    --cp_load_path save/${save_dir}/cp_1000.pth
done
