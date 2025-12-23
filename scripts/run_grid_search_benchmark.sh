#!/bin/bash

cd ..

# # lorenz 63
# dataset="lorenz63"
# sigma_y=1
# seed=42
# methods=("EnKF" "ESRF" "iEnKS-PertObs" "iEnKS-Sqrt" "iEnKS-Order1") 

# # --- Evaluation Loop ---
# for N in 5 10 15 20 40 60 100; do
#     for method in "${methods[@]}"; do
#         echo "Running evaluation for N=$N and method=$method"
#         python grid_search_benchmark.py \
#             --device cpu \
#             --dataset "$dataset" \
#             --N "$N" \
#             --sigma_y "$sigma_y" \
#             --seed "$seed" \
#             --v "$method" \
#             --no_localization \
#             --normal_output \
#             --pf_verification \
#             --pf_N 1000000
#     done
# done

# Rossler
# dataset="rossler"
# sigma_y=1
# seed=42
# methods=("EnKF" "ESRF" "iEnKS-PertObs" "iEnKS-Sqrt" "iEnKS-Order1") 

# # --- Evaluation Loop ---
# for N in 5 10 15 20 40 60 100; do
#     for method in "${methods[@]}"; do
#         echo "Running evaluation for N=$N and method=$method"
#         python grid_search_benchmark.py \
#             --device cpu \
#             --dataset "$dataset" \
#             --N "$N" \
#             --sigma_y "$sigma_y" \
#             --seed "$seed" \
#             --v "$method" \
#             --no_localization \
#             --normal_output 
#     done
# done

# Lorenz 96
dataset="lorenz96"
sigma_y=0.7
seed=42
# methods=("iEnKS-PertObs" "iEnKS-Sqrt" "iEnKS-Order1" "LETKF" "EnKF") 
methods=("iEnKS-Sqrt") 

# --- Evaluation Loop ---
for N in 5 10 15 20 40 60 100; do
    for method in "${methods[@]}"; do
        echo "Running evaluation for N=$N and method=$method"
        python grid_search_benchmark.py \
            --device cpu \
            --dataset "$dataset" \
            --N "$N" \
            --sigma_y "$sigma_y" \
            --seed "$seed" \
            --v "$method" \
            --normal_output 
    done
done
