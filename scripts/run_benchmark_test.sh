#!/bin/bash

cd ..

# Rossler
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
# for N in 5 10 15 20 40 60 100; do
for N in 10; do
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
            --pf_N 1000000
    done
done

# # lorenz 96
# dataset="lorenz96"
# # methods=("EnKF" "ESRF" "iEnKS-PertObs" "iEnKS-Sqrt" "iEnKS-Order1") 
# # methods=("EnKF" "LETKF") 
# methods=("EnKF") 

# sigma_y=1
# # --- Evaluation Loop ---
# for N in 5 10 15 20 40 60 100; do
#     for method in "${methods[@]}"; do
#         echo "Running evaluation for N=$N and method=$method"
#         python evaluate_benchmark.py \
#             --device cuda \
#             --dataset "$dataset" \
#             --N "$N" \
#             --sigma_y "$sigma_y" \
#             --seed 42 \
#             --v "$method" \
#             --normal_output 
#             # --pf_verification \
#             # --pf_N 100000 \
#             # --no_localization
#     done
# done

# sigma_y=0.7
# # --- Evaluation Loop ---
# for N in 5 10 15 20 40 60 100; do
#     for method in "${methods[@]}"; do
#         echo "Running evaluation for N=$N and method=$method"
#         python evaluate_benchmark.py \
#             --device cuda \
#             --dataset "$dataset" \
#             --N "$N" \
#             --sigma_y "$sigma_y" \
#             --seed 42 \
#             --v "$method" \
#             --normal_output 
#             # --pf_verification \
#             # --pf_N 100000 \
#             # --no_localization
#     done
# done

# # methods=("ESRF" "iEnKS-PertObs") 
# methods=("ESRF") 

# sigma_y=1
# # --- Evaluation Loop ---
# for N in 5 10 15 20 40 60 100; do
#     for method in "${methods[@]}"; do
#         echo "Running evaluation for N=$N and method=$method"
#         python evaluate_benchmark.py \
#             --device cuda \
#             --dataset "$dataset" \
#             --N "$N" \
#             --sigma_y "$sigma_y" \
#             --seed 42 \
#             --v "$method" \
#             --normal_output \
#             --no_localization
#     done
# done

# sigma_y=0.7
# # --- Evaluation Loop ---
# for N in 5 10 15 20 40 60 100; do
#     for method in "${methods[@]}"; do
#         echo "Running evaluation for N=$N and method=$method"
#         python evaluate_benchmark.py \
#             --device cuda \
#             --dataset "$dataset" \
#             --N "$N" \
#             --sigma_y "$sigma_y" \
#             --seed 42 \
#             --v "$method" \
#             --normal_output \
#             --no_localization
#     done
# done

# # # KS
# dataset="ks"
# # methods=("EnKF" "ESRF" "iEnKS-PertObs" "iEnKS-Sqrt" "iEnKS-Order1") 
# # methods=("EnKF" "LETKF")
# methods=("EnKF")  

# sigma_y=1
# # --- Evaluation Loop ---
# for N in 5 10 15 20 40 60 100; do
#     for method in "${methods[@]}"; do
#         echo "Running evaluation for N=$N and method=$method"
#         python evaluate_benchmark.py \
#             --device cuda \
#             --dataset "$dataset" \
#             --N "$N" \
#             --sigma_y "$sigma_y" \
#             --seed 42 \
#             --v "$method" \
#             --normal_output 
#             # --pf_verification \
#             # --pf_N 100000 \
#             # --no_localization
#     done
# done

# sigma_y=0.7
# # --- Evaluation Loop ---
# for N in 5 10 15 20 40 60 100; do
#     for method in "${methods[@]}"; do
#         echo "Running evaluation for N=$N and method=$method"
#         python evaluate_benchmark.py \
#             --device cpu \
#             --dataset "$dataset" \
#             --N "$N" \
#             --sigma_y "$sigma_y" \
#             --seed 42 \
#             --v "$method" \
#             --normal_output 
#             # --pf_verification \
#             # --pf_N 100000 \
#             # --no_localization
#     done
# done

# methods=("ESRF" "iEnKS-PertObs") 
# # methods=("ESRF") 

# sigma_y=1
# # --- Evaluation Loop ---
# for N in 5 10 15 20 40 60 100; do
#     for method in "${methods[@]}"; do
#         echo "Running evaluation for N=$N and method=$method"
#         python evaluate_benchmark.py \
#             --device cpu \
#             --dataset "$dataset" \
#             --N "$N" \
#             --sigma_y "$sigma_y" \
#             --seed 42 \
#             --v "$method" \
#             --normal_output \
#             --no_localization
#     done
# done

# sigma_y=0.7
# # --- Evaluation Loop ---
# for N in 5 10 15 20 40 60 100; do
#     for method in "${methods[@]}"; do
#         echo "Running evaluation for N=$N and method=$method"
#         python evaluate_benchmark.py \
#             --device cpu \
#             --dataset "$dataset" \
#             --N "$N" \
#             --sigma_y "$sigma_y" \
#             --seed 42 \
#             --v "$method" \
#             --normal_output \
#             --no_localization
#     done
# done


