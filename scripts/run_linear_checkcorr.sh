#!/bin/bash

cd ..

python train_v2.py \
    --save_epoch 5 \
    --epochs 1000 \
    --dataset linear \
    --N 10 \
    --seed 42 \
    --v CorrTerms \
    --no_localization \
    --no_running_loss \
    --loss_type nl2 

python train_v2.py \
    --save_epoch 5 \
    --epochs 1000 \
    --dataset linear \
    --N 10 \
    --seed 42 \
    --v CorrTerms \
    --no_localization \
    --no_running_loss \
    --loss_type nl2 \
    --weight_decay 1e-2

python train_v2.py \
    --save_epoch 5 \
    --epochs 1000 \
    --dataset linear \
    --N 10 \
    --seed 42 \
    --v CorrTerms \
    --no_localization \
    --no_running_loss \
    --loss_type l2 

python train_v2.py \
    --save_epoch 5 \
    --epochs 1000 \
    --dataset linear \
    --N 10 \
    --seed 42 \
    --v CorrTerms \
    --no_localization \
    --no_running_loss \
    --loss_type l2 \
    --weight_decay 1e-2

# python train_v2.py \
#     --epochs 1000 \
#     --dataset linear \
#     --N 10 \
#     --seed 42 \
#     --v CorrTerms \
#     --no_localization \
#     --no_running_loss \
#     --loss_type nl2 \
#     --es_p 1 \
#     --weight_decay 1e-2 
    
# python train_v2.py \
#     --epochs 1000 \
#     --dataset linear \
#     --N 10 \
#     --seed 42 \
#     --v CorrTerms \
#     --no_localization \
#     --no_running_loss \
#     --loss_type es \
#     --es_p 1 \
#     --weight_decay 1e-2 \
#     --noise_st_input \
#     --mlp_y_type noise_innov \
#     --suffix _nst_nimlp

# python train_v2.py \
#     --epochs 1000 \
#     --dataset linear \
#     --N 10 \
#     --seed 42 \
#     --v CorrTerms \
#     --no_localization \
#     --no_running_loss \
#     --loss_type es \
#     --es_p 1 \
#     --weight_decay 1e-2 \
#     --noise_st_input \
#     --suffix _nst

# python train_v2.py \
#     --epochs 1000 \
#     --dataset linear \
#     --N 10 \
#     --seed 42 \
#     --v CorrTerms \
#     --no_localization \
#     --no_running_loss \
#     --loss_type es \
#     --es_p 1 \
#     --weight_decay 1e-2 \
#     --mlp_y_type noise_innov \
#     --suffix _nimlp