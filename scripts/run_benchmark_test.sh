#!/bin/bash

cd ..

PYTHON_BIN="/home/bhchen/miniconda3/bin/python"

# evaluate_benchmark.py now defaults to loading best infl/loc from
# save/torch_grid_search/<dataset>.csv by matching the full experiment setting.
# When --adaptive_sigma_y is enabled, the lookup uses the adapted sigma_y value.

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
# Keep this aligned with DATASET_INFO['lorenz63']['sigma_y'] when using --adaptive_sigma_y,
# so benchmark/evaluate pipelines adapt to the same final sigma_y.
sigma_y=2.0
seed=42
plot_start_step=100
plot_interval_step=100
plot_end_step=500
methods=("iEnKS-PertObs") 
# methods=("EnKF" "ESRF" "iEnKS-PertObs" "iEnKS-Sqrt" "iEnKS-Order1")
obs_fns=("square") 
# obs_fns=("identity" "arctan")

# --- Evaluation Loop ---
for obs_fn in "${obs_fns[@]}"; do
    for N in 5 10 15 20 40 60 100; do
    # for N in 100; do
        for method in "${methods[@]}"; do
            echo "Running evaluation for obs_fn=$obs_fn, N=$N and method=$method"
            "$PYTHON_BIN" evaluate_benchmark.py \
                --device cuda \
                --dataset "$dataset" \
                --N "$N" \
                --sigma_y "$sigma_y" \
                --seed "$seed" \
                --v "$method" \
                --no_localization \
                --normal_output \
                --pf_verification \
                --pf_N 1000000 \
                --adaptive_sigma_y \
                --obs_fn "$obs_fn" \
                --test_snapshot_start_step "$plot_start_step" \
                --test_snapshot_interval "$plot_interval_step" \
                --test_snapshot_end_step "$plot_end_step" \
                # --save_test_figures 
        done
    done
done

# doubling1d
# dataset="doubling1d"
# # methods=("EnKF" "ESRF" "iEnKS-PertObs" "iEnKS-Sqrt" "iEnKS-Order1")
# # methods=("iEnKS-PertObs" "ESRF")
# methods=("EnKF")

# sigma_y=0.2
# seed=42
# plot_start_step=1
# plot_interval_step=1
# plot_end_step=200
# # --- Evaluation Loop ---
# for N in 1000; do
#     for method in "${methods[@]}"; do
#         echo "Running evaluation for N=$N and method=$method"
#         "$PYTHON_BIN" evaluate_benchmark.py \
#             --device cpu \
#             --dataset "$dataset" \
#             --N "$N" \
#             --sigma_y "$sigma_y" \
#             --seed 42 \
#             --v "$method" \
#             --obs_fn cos2pi \
#             --normal_output \
#             --pf_verification \
#             --pf_N 1000000 \
#             --no_localization \
#             --save_test_figures \
#             --test_snapshot_start_step "$plot_start_step" \
#             --test_snapshot_interval "$plot_interval_step" \
#             --test_snapshot_end_step "$plot_end_step"
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
