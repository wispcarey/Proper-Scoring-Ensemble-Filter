#!/bin/bash

cd ..

# python train_v2.py \
#     --epochs 0 \
#     --dataset linear \
#     --N 10 \
#     --seed 42 \
#     --v EnKF \
#     --no_localization \
#     --normal_output

python train.py \
    --epochs 0 \
    --dataset lorenz63 \
    --sigma_y 1 \
    --N 10 \
    --seed 42 \
    --v EnKF \
    --test_steps 500 \
    --no_localization \
    --normal_output \
    --pf_verification \
    --pf_N 1000000 \
    --sigma_reg None 