#!/bin/bash

cd ..

python train_v2.py \
    --epochs 1000 \
    --dataset linear \
    --N 10 \
    --seed 3 \
    --v CorrTerms \
    --no_localization \
    --loss_type es \
    --es_p 1 \
    --learning_rate 1e-3 \
    --weight_decay 1e-2 \
    --normal_output \
    --noise_st_input \
    --mlp_y_type innov \
    --suffix _nst_and_mlpinnov