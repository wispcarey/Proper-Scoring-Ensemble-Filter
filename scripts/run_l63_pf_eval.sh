#!/bin/bash

cd ..

# lorenz 63
dataset="lorenz63"

sigma_y=1
seed=42

# # corrterms es1
# for N in 5 10 15 20 40 60 100; do
#     python evaluate.py \
#     --dataset $dataset \
#     --N $N \
#     --sigma_y $sigma_y \
#     --seed $seed \
#     --v CorrTerms \
#     --no_localization \
#     --normal_output \
#     --test_steps 500 \
#     --pf_verification \
#     --pf_N 1000000 \
#     --sigma_reg None \
#     --cp_load_path save/2025-06-28_21-55lorenz63_1.0_10_60_8192_es_joint_CorrTerms/cp_1000.pth
# done

# # corrterms l2
# for N in 5 10 15 20 40 60 100; do
#     python evaluate.py \
#     --dataset $dataset \
#     --N $N \
#     --sigma_y $sigma_y \
#     --seed $seed \
#     --v CorrTerms \
#     --no_localization \
#     --normal_output \
#     --test_steps 500 \
#     --pf_verification \
#     --pf_N 1000000 \
#     --sigma_reg None \
#     --cp_load_path save/2025-06-28_21-52lorenz63_1.0_10_60_8192_l2_joint_CorrTerms/cp_1000.pth
# done


# EtE-LRes es1
# for N in 5 10 15 20 40 60 100; do
#     python evaluate.py \
#     --dataset $dataset \
#     --N $N \
#     --sigma_y $sigma_y \
#     --seed $seed \
#     --v EtE-LRes \
#     --no_localization \
#     --normal_output \
#     --test_steps 500 \
#     --pf_verification \
#     --pf_N 1000000 \
#     --sigma_reg None \
#     --cp_load_path save/EtE-LinearCheck/2025-07-06_12-24lorenz63_1.0_10_60_8192_es_joint_EtE-LRes/cp_1000.pth
# done

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
    --pf_N 1000000 \
    --sigma_reg None \
    --cp_load_path save/EtE-LinearCheck/2025-07-09_19-08lorenz63_1.0_10_60_8192_es_joint_EtE-LRes_nst/cp_1000.pth \
    --noise_st_input 
done

# EtE-LRes es1 nmlp
# for N in 5 10 15 20 40 60 100; do
#     python evaluate.py \
#     --dataset $dataset \
#     --N $N \
#     --sigma_y $sigma_y \
#     --seed $seed \
#     --v EtE-LRes \
#     --no_localization \
#     --normal_output \
#     --test_steps 500 \
#     --pf_verification \
#     --pf_N 1000000 \
#     --sigma_reg None \
#     --cp_load_path save/EtE-LinearCheck/2025-07-09_23-25lorenz63_1.0_10_60_8192_es_joint_EtE-LRes_nimlp/cp_1000.pth \
#     --mlp_y_type noise_innov 
# done

# # EtE-LRes es1 nst & nmlp
# for N in 5 10 15 20 40 60 100; do
#     python evaluate.py \
#     --dataset $dataset \
#     --N $N \
#     --sigma_y $sigma_y \
#     --seed $seed \
#     --v EtE-LRes \
#     --no_localization \
#     --normal_output \
#     --test_steps 500 \
#     --pf_verification \
#     --pf_N 1000000 \
#     --sigma_reg None \
#     --cp_load_path save/EtE-LinearCheck/2025-07-09_23-25lorenz63_1.0_10_60_8192_es_joint_EtE-LRes_nst_nimlp/cp_1000.pth \
#     --noise_st_input \
#     --mlp_y_type noise_innov 
# done

# ete-lres es1

# python evaluate.py \
# --dataset $dataset \
# --N 5 \
# --sigma_y $sigma_y \
# --seed $seed \
# --v EtE-LRes \
# --no_localization \
# --normal_output \
# --test_steps 500 \
# --pf_verification \
# --pf_N 1000000 \
# --sigma_reg None \
# --cp_load_path save/2025-07-06_16-00lorenz63_1.0_5_60_8192_es_joint_EtE-LRes/cp_500.pth

# python evaluate.py \
# --dataset $dataset \
# --N 10 \
# --sigma_y $sigma_y \
# --seed $seed \
# --v EtE-LRes \
# --no_localization \
# --normal_output \
# --test_steps 500 \
# --pf_verification \
# --pf_N 1000000 \
# --sigma_reg None \
# --cp_load_path save/2025-07-06_12-24lorenz63_1.0_10_60_8192_es_joint_EtE-LRes/cp_500.pth

# python evaluate.py \
# --dataset $dataset \
# --N 15 \
# --sigma_y $sigma_y \
# --seed $seed \
# --v EtE-LRes \
# --no_localization \
# --normal_output \
# --test_steps 500 \
# --pf_verification \
# --pf_N 1000000 \
# --sigma_reg None \
# --cp_load_path save/2025-07-06_16-00lorenz63_1.0_15_60_8192_es_joint_EtE-LRes/cp_500.pth


# ete es2
# save_dir="2025-06-28_21-52lorenz63_1.0_10_60_8192_l2_joint_EtE"
# for N in 10; do
#     python evaluate.py \
#     --dataset $dataset \
#     --N $N \
#     --sigma_y $sigma_y \
#     --seed $seed \
#     --v EtE \
#     --no_localization \
#     --normal_output \
#     --test_steps 500 \
#     --pf_verification \
#     --pf_N 1000000 \
#     --sigma_reg None \
#     --cp_load_path save/${save_dir}/cp_1000.pth
# done

# # ete es1
# save_dir="2025-06-28_21-55lorenz63_1.0_10_60_8192_es_joint_EtE2"
# for N in 10; do
#     python evaluate.py \
#     --dataset $dataset \
#     --N $N \
#     --sigma_y $sigma_y \
#     --seed $seed \
#     --v EtE2 \
#     --no_localization \
#     --normal_output \
#     --test_steps 500 \
#     --pf_verification \
#     --pf_N 1000000 \
#     --sigma_reg None \
#     --cp_load_path save/${save_dir}/cp_1000.pth
# done

# # ete es2
# save_dir="2025-06-28_21-52lorenz63_1.0_10_60_8192_l2_joint_EtE2"
# for N in 10; do
#     python evaluate.py \
#     --dataset $dataset \
#     --N $N \
#     --sigma_y $sigma_y \
#     --seed $seed \
#     --v EtE2 \
#     --no_localization \
#     --normal_output \
#     --test_steps 500 \
#     --pf_verification \
#     --pf_N 1000000 \
#     --sigma_reg None \
#     --cp_load_path save/${save_dir}/cp_1000.pth
# done
