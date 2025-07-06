#!/bin/bash

cd ..

python train.py \
    --dataset lorenz63 \
    --epochs 1000 \
    --N 10 \
    --sigma_y 1 \
    --seed 42 \
    --v EtE-LRes \
    --no_localization \
    --loss_type es \
    --es_p 1 \
    --test_steps 500 \
    --normal_output

# python train.py \
#     --dataset lorenz63 \
#     --epochs 500 \
#     --N 10 \
#     --sigma_y 1 \
#     --seed 42 \
#     --v CorrTerms \
#     --no_localization \
#     --loss_type nl2 \
#     --normal_output \
#     --test_steps 100 \
#     --pf_verification \
#     --pf_N 2000 \
#     --sigma_reg 0

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