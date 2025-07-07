#!/bin/bash

cd ..

python train_v2.py \
    --epochs 1000 \
    --dataset linear \
    --N 10 \
    --seed 42 \
    --v CorrTerms \
    --no_localization \
    --loss_type es \
    --es_p 1 \
    --no_running_loss \
    --weight_decay 1e-1 