#!/bin/bash

cd ..

python train.py \
   --dataset lorenz63 \
   --epochs 1000 \
   --N 10 \
   --sigma_y 1 \
   --seed 42 \
   --v EtE-LRes \
   --no_localization \
   --loss_type es \
   --es_p 1 \
   --test_steps 500 \
   --pf_verification \
   --pf_N 1000000 \
   --sigma_reg None