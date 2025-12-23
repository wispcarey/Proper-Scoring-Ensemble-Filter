#!/bin/bash

cd ..

dataset="linear"

sigma_y=1
seed=42

# EnKF
for N in 4 6 8 10; do
    python evaluate_linear.py \
    --dataset $dataset \
    --N $N \
    --sigma_y $sigma_y \
    --seed $seed \
    --v EnKF \
    --no_localization \
    --normal_output \
    --test_steps 100 
done

# corrterms es1
for N in 4 6 8 10; do
    python evaluate_linear.py \
    --dataset $dataset \
    --N $N \
    --sigma_y $sigma_y \
    --seed $seed \
    --v CorrTerms \
    --no_localization \
    --normal_output \
    --test_steps 100 \
    --cp_load_path save/linear_es_vs_l2/2025-08-14_20-05linear_1_10_60_8192_es_joint_CorrTerms/cp_1000.pth
done

# corrterms l2
for N in 4 6 8 10; do
    python evaluate_linear.py \
    --dataset $dataset \
    --N $N \
    --sigma_y $sigma_y \
    --seed $seed \
    --v CorrTerms \
    --no_localization \
    --normal_output \
    --test_steps 100 \
    --cp_load_path save/linear_es_vs_l2/2025-08-14_20-04linear_1_10_60_8192_l2_joint_CorrTerms/cp_1000.pth
done

# EtE-LRes es1 nst
for N in 4 6 8 10; do
    python evaluate_linear.py \
    --dataset $dataset \
    --N $N \
    --sigma_y $sigma_y \
    --seed $seed \
    --v EtE-LRes \
    --no_localization \
    --normal_output \
    --test_steps 100 \
    --cp_load_path save/linear_es_vs_l2/2025-08-14_20-04linear_1_10_60_8192_l2_joint_EtE-LRes_nst/cp_1000.pth \
    --noise_st_input 
done

# EtE-LRes es1 nst
for N in 2 4 6 8 10; do
    python evaluate_linear.py \
    --dataset $dataset \
    --N $N \
    --sigma_y $sigma_y \
    --seed $seed \
    --v EtE-LRes \
    --no_localization \
    --normal_output \
    --test_steps 100 \
    --cp_load_path save/linear_es_vs_l2/2025-08-14_19-59linear_1_10_60_8192_es_joint_EtE-LRes_nst/cp_1000.pth \
    --noise_st_input 
done
