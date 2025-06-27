#!/bin/bash

cd ..

python train.py \
    --dataset lorenz63 \
    --epochs 500 \
    --N 10 \
    --sigma_y 1 \
    --seed 42 \
    --v CorrTerms \
    --no_localization \
    --loss_type nl2 \
    --normal_output \
    --test_steps 60 \
    --pf_verification \
    --pf_N 2001 \
    --sigma_reg 1e-1

# python train.py \
#     --dataset lorenz63 \
#     --epochs 500 \
#     --N 10 \
#     --sigma_y 1 \
#     --seed 42 \
#     --v CorrTerms \
#     --no_localization \
#     --loss_type nes \
#     --es_p 1 \
#     --normal_output \
#     --test_steps 60 \
#     --pf_verification \
#     --pf_N 10000

# python train.py \
#     --dataset lorenz63 \
#     --epochs 500 \
#     --N 10 \
#     --sigma_y 1 \
#     --seed 42 \
#     --v EtE \
#     --no_localization \
#     --loss_type nl2 \
#     --normal_output \
#     --test_steps 60 \
#     --pf_verification \
#     --pf_N 10000

# python train.py \
#     --dataset lorenz63 \
#     --epochs 500 \
#     --N 10 \
#     --sigma_y 1 \
#     --seed 42 \
#     --v EtE \
#     --no_localization \
#     --loss_type nes \
#     --es_p 1 \
#     --normal_output \
#     --test_steps 60 \
#     --pf_verification \
#     --pf_N 10000

# python train.py \
#     --dataset lorenz63 \
#     --epochs 500 \
#     --N 10 \
#     --sigma_y 1 \
#     --seed 42 \
#     --v EtE2 \
#     --no_localization \
#     --loss_type nl2 \
#     --normal_output \
#     --test_steps 60 \
#     --pf_verification \
#     --pf_N 10000

# python train.py \
#     --dataset lorenz63 \
#     --epochs 500 \
#     --N 10 \
#     --sigma_y 1 \
#     --seed 42 \
#     --v EtE2 \
#     --no_localization \
#     --loss_type nes \
#     --es_p 1 \
#     --normal_output \
#     --test_steps 60 \
#     --pf_verification \
#     --pf_N 10000