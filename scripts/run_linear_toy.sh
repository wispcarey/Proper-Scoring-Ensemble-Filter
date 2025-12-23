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
    --dim 10 \
    --obs_dim 5

python train_v2.py \
    --save_epoch 5 \
    --epochs 100 \
    --dataset linear \
    --N 10 \
    --seed 42 \
    --v EtE-LRes \
    --no_localization \
    --no_running_loss \
    --loss_type nl2 \
    --dim 10 \
    --obs_dim 5