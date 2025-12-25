#!/bin/bash

cd ..

python train_v2.py \
    --save_epoch 5 \
    --epochs 100 \
    --dataset linear \
    --N 10 \
    --seed 42 \
    --v EtE-LRes \
    --no_localization \
    --no_running_loss \
    --loss_type es \
    --es_p 1 \
    --dim 20 \
    --obs_dim 10 \
    --suffix 20

# python train_v2.py \
#     --save_epoch 5 \
#     --epochs 100 \
#     --dataset linear \
#     --N 10 \
#     --seed 42 \
#     --v EtE-LRes \
#     --no_localization \
#     --no_running_loss \
#     --loss_type l2 \
#     --dim 20 \
#     --obs_dim 10 \
#     --suffix 20