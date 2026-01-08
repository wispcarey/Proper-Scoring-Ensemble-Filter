#!/bin/bash

cd ..

python evaluate.py \
   --dataset lorenz63 \
   --N 10 \
   --sigma_y 1 \
   --seed 42 \
   --v EnKF \
   --no_localization \
   --no_running_loss \
   --test_steps 500 \
   --pf_verification \
   --pf_N 1000000 \
   --sigma_reg None