#!/bin/bash

cd ..

# # Rossler
# dataset="rossler"
# sigma_y=1
# seed=42
# methods=("EnKF" "ESRF" "iEnKS-PertObs" "iEnKS-Sqrt" "iEnKS-Order1") 

# # --- Evaluation Loop ---
# for N in 5 10 15 20 40 60 100; do
#     for method in "${methods[@]}"; do
#         echo "Running evaluation for N=$N and method=$method"
#         python evaluate_benchmark.py \
#             --device cpu \
#             --dataset "$dataset" \
#             --N "$N" \
#             --sigma_y "$sigma_y" \
#             --seed "$seed" \
#             --v "$method" \
#             --no_localization \
#             --normal_output \
#             --pf_verification \
#             --pf_N 100000
#     done
# done

# lorenz 63
dataset="lorenz63"
sigma_y=1
seed=42
methods=("EnKF" "ESRF" "iEnKS-PertObs" "iEnKS-Sqrt" "iEnKS-Order1") 

# --- Evaluation Loop ---
for N in 5 10 15 20 40 60 100; do
    for method in "${methods[@]}"; do
        echo "Running evaluation for N=$N and method=$method"
        python evaluate_benchmark.py \
            --device cpu \
            --dataset "$dataset" \
            --N "$N" \
            --sigma_y "$sigma_y" \
            --seed "$seed" \
            --v "$method" \
            --no_localization \
            --normal_output \
            --pf_verification \
            --pf_N 100000
    done
done

# # lorenz 96
# dataset="lorenz96"
# sigma_y=1
# seed=42
# methods=("EnKF" "ESRF" "iEnKS-PertObs" "iEnKS-Sqrt" "iEnKS-Order1") 

# # --- Evaluation Loop ---
# for N in 5 10 15 20 40; do
#     for method in "${methods[@]}"; do
#         echo "Running evaluation for N=$N and method=$method"
#         python evaluate_benchmark.py \
#             --device cpu \
#             --dataset "$dataset" \
#             --N "$N" \
#             --sigma_y "$sigma_y" \
#             --seed "$seed" \
#             --v "$method" \
#             --normal_output \
#             --pf_verification \
#             --pf_N 100000 \
#             --no_localization
#     done
# done


