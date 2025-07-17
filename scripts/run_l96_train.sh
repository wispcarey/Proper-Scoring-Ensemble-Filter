#!/bin/bash

cd ..

python train.py \
    --dataset lorenz96 \
    --epochs 1000 \
    --N 10 \
    --sigma_y 1 \
    --seed 42 \
    --v EtE-LRes \
    --loss_type es \
    --es_p 1 \
    --test_steps 500 \
    --normal_output \
    --v EtE-LRes \
    --no_localization \
    --no_running_loss \
    --mlp_y_type noise_innov \
    --suffix _nimlp





