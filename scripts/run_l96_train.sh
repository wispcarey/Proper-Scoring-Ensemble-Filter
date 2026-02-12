#!/bin/bash

cd ..

python train.py \
   --dataset lorenz96 \
   --epochs 300 \
   --N 10 \
   --sigma_y 1.0 \
   --seed 42 \
   --v EtE-LRes \
   --no_running_loss \
   --weight_decay 0.0 \
   --loss_type es \
   --es_p 1 \
   --save_epoch 10 \
   --obs_fn square \
   --normal_output \
   --learning_rate 5e-3 \
   --adaptive_sigma_y