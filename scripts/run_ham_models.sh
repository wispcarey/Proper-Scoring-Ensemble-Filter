#!/bin/bash

cd ..

python train.py \
    --epochs 25 \
    --dataset circle \
    --N 10 \
    --seed 42 \
    --v CorrTerms \
    --no_localization \
    --loss_type es \
    --es_p 1 \
    --no_running_loss \
    --noise_st_input \
    --mlp_y_type noise_innov \
    --suffix _nst_nimlp

python train.py \
    --epochs 25 \
    --dataset Hdoublewell \
    --N 10 \
    --seed 42 \
    --v CorrTerms \
    --no_localization \
    --loss_type es \
    --es_p 1 \
    --no_running_loss \
    --noise_st_input \
    --mlp_y_type noise_innov \
    --suffix _nst_nimlp