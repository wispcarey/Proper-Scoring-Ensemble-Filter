#!/bin/bash

cd ..

dataset="linear"

sigma_y=1
seed=42

# EnKF
python grid_search_linear.py \
    --dataset $dataset \
    --N 10 \
    --sigma_y $sigma_y \
    --seed $seed \
    --v LETKF \
    --normal_output \
    --test_steps 100 \
    --dim 20 \
    --obs_dim 10