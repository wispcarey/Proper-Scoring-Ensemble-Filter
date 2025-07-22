#!/bin/bash

cd ..
python evaluate_linear_uncertainty.py \
    --dataset linear \
    --N 2 \
    --seed 42 \
    --normal_output

python evaluate_linear_uncertainty.py \
    --dataset linear \
    --N 5 \
    --seed 42 \
    --normal_output

python evaluate_linear_uncertainty.py \
    --dataset linear \
    --N 10 \
    --seed 42 \
    --normal_output

