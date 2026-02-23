#!/bin/bash

cd ..

python train.py \
   --dataset lorenz63 \
   --epochs 300 \
   --N 10 \
   --sigma_y 1.0 \
   --seed 42 \
   --v EtE-LRes \
   --no_localization \
   --no_running_loss \
   --loss_type es \
   --es_p 1 \
   --test_steps 500 \
   --sigma_reg None \
   --save_epoch 10 \
   --obs_fn square \
   --save_test_figures