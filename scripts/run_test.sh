#!/bin/bash

cd ..

# Rossler
dataset="rossler"

sigma_y=1
seed=42
save_dir="rossler_EtE_LRes/2025-08-07_16-44rossler_1.0_10_60_8192_es_joint_EtE-LRes"

# sigma_y = $sigma_y, EnST
for N in 5 10 15 20 40 60 100; do
    python evaluate.py \
        --v EtE-LRes \
        --dataset $dataset \
        --N $N \
        --sigma_y $sigma_y \
        --seed $seed \
        --no_localization \
        --cp_load_path save/${save_dir}/cp_1000.pth \
        --pf_verification \
        --pf_N 10000 \
        --sigma_reg None 
done

dataset="rossler"

sigma_y=1
seed=42
save_dir="rossler_EtE_LRes/2025-08-07_16-44rossler_1.0_10_60_8192_es_joint_EtE-LRes_nst"

# sigma_y = $sigma_y, EnST
for N in 5 10 15 20 40 60 100; do
    python evaluate.py \
        --v EtE-LRes \
        --dataset $dataset \
        --N $N \
        --sigma_y $sigma_y \
        --seed $seed \
        --no_localization \
        --cp_load_path save/${save_dir}/cp_1000.pth \
        --pf_verification \
        --pf_N 10000 \
        --sigma_reg None 
done

# # lorenz 63
# dataset="lorenz63"

# sigma_y=1
# seed=42
# save_dir="lorenz63_EtE_LRes/2025-07-09_19-08lorenz63_1.0_10_60_8192_es_joint_EtE-LRes_nst"

# # sigma_y = $sigma_y, EnST
# for N in 5 10 15 20 40 60 100; do
#     python evaluate.py \
#         --v EtE-LRes \
#         --dataset $dataset \
#         --N $N \
#         --sigma_y $sigma_y \
#         --seed $seed \
#         --no_localization \
#         --cp_load_path save/${save_dir}/cp_1000.pth \
#         --pf_verification \
#         --pf_N 1000000 \
#         --sigma_reg None 
# done

# sigma_y=0.7
# seed=42
# save_dir="2025-04-10_19-40lorenz63_0.7_10_60_8192_norm_EnST_joint"

# # sigma_y = $sigma_y, EnST
# for N in 5 10 15 20 40 60 100; do
#     python evaluate.py \
#         --dataset $dataset \
#         --N $N \
#         --sigma_y $sigma_y \
#         --seed $seed \
#         --no_localization \
#         --cp_load_path save/${save_dir}/cp_1000.pth
# done

# # lorenz 96
# dataset="lorenz96"

# sigma_y=1
# seed=42
# save_dir="2025-04-10_09-31lorenz96_1.0_10_60_8192_norm_EnST_joint"

# # sigma_y = $sigma_y, EnST
# for N in 5 10 15 20 40 60 100; do
#     python evaluate.py \
#         --dataset $dataset \
#         --N $N \
#         --sigma_y $sigma_y \
#         --seed $seed \
#         --cp_load_path save/${save_dir}/cp_1000.pth
# done

# sigma_y=0.7
# seed=42
# save_dir="2025-04-10_13-18lorenz96_0.7_10_60_8192_norm_EnST_joint"

# # sigma_y = $sigma_y, EnST
# for N in 5 10 15 20 40 60 100; do
#     python evaluate.py \
#         --dataset $dataset \
#         --N $N \
#         --sigma_y $sigma_y \
#         --seed $seed \
#         --cp_load_path save/${save_dir}/cp_1000.pth
# done

# # ks
# dataset="ks"

# sigma_y=1
# seed=42
# save_dir="2025-04-09_18-18ks_1.0_10_60_8192_norm_EnST_joint"

# # sigma_y = $sigma_y, EnST
# for N in 5 10 15 20 40 60 100; do
#     python evaluate.py \
#         --dataset $dataset \
#         --N $N \
#         --sigma_y $sigma_y \
#         --seed $seed \
#         --cp_load_path save/${save_dir}/cp_1000.pth
# done

# sigma_y=0.7
# seed=42
# save_dir="2025-04-10_01-52ks_0.7_10_60_8192_norm_EnST_joint"

# # sigma_y = $sigma_y, EnST
# for N in 5 10 15 20 40 60 100; do
#     python evaluate.py \
#         --dataset $dataset \
#         --N $N \
#         --sigma_y $sigma_y \
#         --seed $seed \
#         --cp_load_path save/${save_dir}/cp_1000.pth
# done

