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
   --loss_type es \
   --es_p 1 \
   --save_epoch 10 \
   --obs_fn square_root \
   --normal_output