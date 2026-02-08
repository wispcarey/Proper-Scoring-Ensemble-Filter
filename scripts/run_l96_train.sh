#!/bin/bash

cd ..

python train.py \
   --dataset lorenz96 \
   --epochs 300 \
   --N 10 \
   --sigma_y 1.0 \
   --seed 42 \
   --v CorrTerms \
   --no_running_loss \
   --weight_decay 0.01 \
   --loss_type es \
   --es_p 1 \
   --save_epoch 10 \
   --obs_fn square_root \
   --normal_output