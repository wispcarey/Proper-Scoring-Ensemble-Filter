#!/bin/bash

cd ..

############ Linear
# python check_linearity.py \
#     --dataset linear \
#     --seed 42 \
#     --v EtE-LRes \
#     --normal_output \
#     --cp_load_path save/EtE-LinearCheck/2025-07-06_12-24linear_1_10_60_8192_es_joint_EtE-LRes/cp_1000.pth

# python check_linearity.py \
#     --dataset linear \
#     --seed 42 \
#     --v EtE-LRes \
#     --normal_output \
#     --cp_load_path save/EtE-LinearCheck/2025-07-09_23-25linear_1_10_60_8192_es_joint_EtE-LRes_nst/cp_1000.pth

# python check_linearity.py \
#     --dataset linear \
#     --seed 42 \
#     --v EtE-LRes \
#     --normal_output \
#     --cp_load_path save/EtE-LinearCheck/2025-07-09_23-33linear_1_10_60_8192_es_joint_EtE-LRes_nimlp/cp_1000.pth

# python check_linearity.py \
#     --dataset linear \
#     --seed 42 \
#     --v EtE-LRes \
#     --normal_output \
#     --cp_load_path save/EtE-LinearCheck/2025-07-09_23-33linear_1_10_60_8192_es_joint_EtE-LRes_nst_nimlp/cp_1000.pth

############ Lorenz '63
python check_linearity.py \
    --dataset lorenz63 \
    --seed 42 \
    --v EtE-LRes \
    --normal_output \
    --cp_load_path save/EtE-LinearCheck/2025-07-06_12-24lorenz63_1.0_10_60_8192_es_joint_EtE-LRes/cp_1000.pth

python check_linearity.py \
    --dataset lorenz63 \
    --seed 42 \
    --v EtE-LRes \
    --normal_output \
    --cp_load_path save/EtE-LinearCheck/2025-07-09_19-08lorenz63_1.0_10_60_8192_es_joint_EtE-LRes_nst/cp_1000.pth

python check_linearity.py \
    --dataset lorenz63 \
    --seed 42 \
    --v EtE-LRes \
    --normal_output \
    --cp_load_path save/EtE-LinearCheck/2025-07-09_23-25lorenz63_1.0_10_60_8192_es_joint_EtE-LRes_nimlp/cp_1000.pth

python check_linearity.py \
    --dataset lorenz63 \
    --seed 42 \
    --v EtE-LRes \
    --normal_output \
    --cp_load_path save/EtE-LinearCheck/2025-07-09_23-25lorenz63_1.0_10_60_8192_es_joint_EtE-LRes_nst_nimlp/cp_1000.pth