#!/bin/bash

cd ..

python gen_pf_results.py \
    --dataset lorenz63 \
    --sigma_y 1 \
    --seed 42 \
    --normal_output \
    --test_steps 100 \
    --pf_verification \
    --pf_N 10000 \
    --seed 1 \
    --sigma_reg None