#!/bin/bash

cd ..

python train.py \
   --dataset lorenz63 \
   --num_loader_workers 16 \
   --epochs 1000 \
   --N 10 \
   --sigma_y 1 \
   --seed 42 \
   # --v EtE-LRes \
   --v CorrTerms \
   --no_running_loss \
   --loss_type nl2 \
   --no_localization \
   --es_p 1

# python train.py \
#    --dataset rossler \
#    --epochs 1000 \
#    --N 10 \
#    --sigma_y 1 \
#    --seed 42 \
#    --v EtE-LRes \
#    --no_localization \
#    --loss_type es \
#    --es_p 1 \
#    --test_steps 500 \
#    --no_running_loss 

# python train.py \
#     --dataset ks \
#     --N 10 \
#     --sigma_y 1 \
#     --seed 42 

# python train.py \
#     --dataset ks \
#     --N 10 \
#     --sigma_y 0.7 \
#     --seed 42 

# python train.py \
#     --dataset lorenz96 \
#     --N 10 \
#     --sigma_y 1 \
#     --seed 42 \
#     --normal_output

# python train.py \
#     --dataset lorenz96 \
#     --N 10 \
#     --sigma_y 0.7 \
#     --seed 42 

# python train.py \
#     --dataset lorenz63 \
#     --N 10 \
#     --sigma_y 1 \
#     --seed 1 \
#     --no_localization 

# python train.py \
#     --dataset lorenz63 \
#     --N 10 \
#     --sigma_y 0.7 \
#     --seed 1 \
#     --no_localization 





