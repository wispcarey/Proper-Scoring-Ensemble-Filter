#!/bin/bash

cd ..
dataset="linear"

sigma_y=1
seed=42

for N in 10; do
    python evaluate_linear_uncertainty.py \
    --dataset $dataset \
    --N $N \
    --sigma_y $sigma_y \
    --seed $seed \
    --normal_output \
    --dim 10 \
    --obs_dim 5
done