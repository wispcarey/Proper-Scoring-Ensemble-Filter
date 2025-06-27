#!/bin/bash

cd ..

# python train.py \
#     --dataset ks \
#     --N 10 \
#     --sigma_y 1 \
#     --seed 42 

# python train.py \
#     --dataset ks \
#     --N 10 \
#     --sigma_y 0.7 \
#     --seed 42 

python train.py \
    --dataset lorenz96 \
    --N 10 \
    --sigma_y 1 \
    --seed 42 \
    --normal_output

# python train.py \
#     --dataset lorenz96 \
#     --N 10 \
#     --sigma_y 0.7 \
#     --seed 42 

# python train.py \
#     --dataset lorenz63 \
#     --N 10 \
#     --sigma_y 1 \
#     --seed 1 \
#     --no_localization 

# python train.py \
#     --dataset lorenz63 \
#     --N 10 \
#     --sigma_y 0.7 \
#     --seed 1 \
#     --no_localization 





