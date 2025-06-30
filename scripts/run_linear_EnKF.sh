#!/bin/bash

cd ..

python train_v2.py \
    --epochs 0 \
    --dataset linear \
    --N 10 \
    --seed 42 \
    --v EnKF \
    --no_localization \
    --normal_output

# python train_v2.py \
#     --epochs 500 \
#     --dataset linear \
#     --N 10 \
#     --seed 42 \
#     --v CorrTerms \
#     --no_localization \
#     --loss_type nes \
#     --es_p 1 

# python train_v2.py \
#     --epochs 1500 \
#     --dataset linear \
#     --N 10 \
#     --seed 42 \
#     --v EtE \
#     --no_localization \
#     --loss_type es \
#     --es_p 1 \
#     --normal_output

# python train_v2.py \
#     --epochs 500 \
#     --dataset linear \
#     --N 10 \
#     --seed 42 \
#     --v EtE \
#     --no_localization \
#     --loss_type nes \
#     --es_p 1 

# python train_v2.py \
#     --epochs 1500 \
#     --dataset linear \
#     --N 10 \
#     --seed 42 \
#     --v EtE2 \
#     --no_localization \
#     --loss_type es \
#     --es_p 1 

# python train_v2.py \
#     --epochs 500 \
#     --dataset linear \
#     --N 10 \
#     --seed 42 \
#     --v EtE2 \
#     --no_localization \
#     --loss_type nes \
#     --es_p 1 