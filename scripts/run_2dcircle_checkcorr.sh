#!/bin/bash

cd ..

python train.py \
    --epochs 1000 \
    --dataset circle \
    --N 10 \
    --seed 42 \
    --v CorrTerms \
    --no_localization \
    --loss_type es \
    --es_p 1 \
    --weight_decay 1e-2 
    
python train.py \
    --epochs 1000 \
    --dataset circle \
    --N 10 \
    --seed 42 \
    --v CorrTerms \
    --no_localization \
    --loss_type es \
    --es_p 1 \
    --weight_decay 1e-2 \
    --noise_st_input \
    --mlp_y_type noise_innov \
    --suffix _nst_nimlp

python train.py \
    --epochs 1000 \
    --dataset circle \
    --N 10 \
    --seed 42 \
    --v CorrTerms \
    --no_localization \
    --loss_type es \
    --es_p 1 \
    --weight_decay 1e-2 \
    --noise_st_input \
    --suffix _nst

python train.py \
    --epochs 1000 \
    --dataset circle \
    --N 10 \
    --seed 42 \
    --v CorrTerms \
    --no_localization \
    --loss_type es \
    --es_p 1 \
    --weight_decay 1e-2 \
    --mlp_y_type noise_innov \
    --suffix _nimlp