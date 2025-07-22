#!/bin/bash

cd ..

# #W2 = 6.643
# python train_v2.py \
#     --epochs 0 \
#     --dataset linear \
#     --N 2 \
#     --seed 42 \
#     --v EnKF \
#     --no_localization \
#     --normal_output

# #W2 = 3.304
# python train_v2.py \
#     --epochs 0 \
#     --dataset linear \
#     --N 5 \
#     --seed 42 \
#     --v EnKF \
#     --no_localization \
#     --normal_output

# #W2 = 2.055
# python train_v2.py \
#     --epochs 0 \
#     --dataset linear \
#     --N 10 \
#     --seed 42 \
#     --v EnKF \
#     --no_localization \
#     --normal_output

for N in 5 10 15 20 40 60 100; do
    python train.py \
        --epochs 0 \
        --dataset lorenz63 \
        --sigma_y 1 \
        --N $N \
        --seed 42 \
        --v EnKF \
        --test_steps 500 \
        --no_localization \
        --normal_output \
        --pf_verification \
        --pf_N 1000000 \
        --sigma_reg None 
done