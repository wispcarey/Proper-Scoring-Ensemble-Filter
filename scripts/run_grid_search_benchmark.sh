#!/bin/bash

cd ..

seed=42
cpu_workers=16
grid_search_num_seeds=2
ensemble_sizes=(5 10 15 20 40 60 100)

# Lorenz 63
dataset="lorenz63"
methods=("iEnKS-PertObs")
obs_fns=("square")

for obs_fn in "${obs_fns[@]}"; do
    for N in "${ensemble_sizes[@]}"; do
        for method in "${methods[@]}"; do
            echo "Running grid search: dataset=$dataset obs_fn=$obs_fn N=$N method=$method"
            python grid_search_benchmark.py \
                --device cpu \
                --dataset "$dataset" \
                --N "$N" \
                --seed "$seed" \
                --v "$method" \
                --obs_fn "$obs_fn" \
                --adaptive_sigma_y \
                --grid_search_cpu_workers "$cpu_workers" \
                --grid_search_num_seeds "$grid_search_num_seeds" \
                --disable_tqdm \
                --no_localization \
                --normal_output \
                --pf_verification \
                --pf_N 1000000
        done
    done
done

# Doubling1D
dataset="doubling1d"
methods=("EnKF" "ESRF" "iEnKS-PertObs")
obs_fns=("cos2pi")

for obs_fn in "${obs_fns[@]}"; do
    for N in "${ensemble_sizes[@]}"; do
        for method in "${methods[@]}"; do
            echo "Running grid search: dataset=$dataset obs_fn=$obs_fn N=$N method=$method"
            python grid_search_benchmark.py \
                --device cpu \
                --dataset "$dataset" \
                --N "$N" \
                --seed "$seed" \
                --v "$method" \
                --obs_fn "$obs_fn" \
                --adaptive_sigma_y \
                --grid_search_cpu_workers "$cpu_workers" \
                --grid_search_num_seeds "$grid_search_num_seeds" \
                --disable_tqdm \
                --no_localization \
                --normal_output \
                --pf_verification \
                --pf_N 1000000
        done
    done
done

# Lorenz 96
# dataset="lorenz96"
# methods=("EnKF" "ESRF" "iEnKS-PertObs" "iEnKS-Sqrt" "iEnKS-Order1" "LETKF")
# obs_fns=("identity")
#
# for obs_fn in "${obs_fns[@]}"; do
#     for N in "${ensemble_sizes[@]}"; do
#         for method in "${methods[@]}"; do
#             echo "Running grid search: dataset=$dataset obs_fn=$obs_fn N=$N method=$method"
#             python grid_search_benchmark.py \
#                 --device cpu \
#                 --dataset "$dataset" \
#                 --N "$N" \
#                 --seed "$seed" \
#                 --v "$method" \
#                 --obs_fn "$obs_fn" \
#                 --adaptive_sigma_y \
#                 --grid_search_cpu_workers "$cpu_workers" \
#                 --grid_search_num_seeds "$grid_search_num_seeds" \
#                 --disable_tqdm \
#                 --normal_output
#         done
#     done
# done
