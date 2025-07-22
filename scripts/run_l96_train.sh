#!/bin/bash

cd ..

python train.py \
    --dataset lorenz96 \
    --epochs 1000 \
    --N 10 \
    --sigma_y 1 \
    --seed 42 \
    --v CorrTerms \
    --loss_type es \
    --es_p 1 \
    --test_steps 500 \
    --no_running_loss 





