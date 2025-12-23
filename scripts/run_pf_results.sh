#!/bin/bash

cd ..

echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Python: $(which python)"
echo "----------------------------------------------------"

echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Python: $(which python)"
echo "----------------------------------------------------"

SEEDS=(42)
PARTICLE_NUMBERS=(1000000)

for seed_val in "${SEEDS[@]}"; do
    for pf_n_val in "${PARTICLE_NUMBERS[@]}"; do
        
        echo "============================================================"
        echo "Running with Seed: $seed_val and Particle Count (pf_N): $pf_n_val"
        echo "============================================================"
        
        python gen_pf_results.py \
            --dataset lorenz63 \
            --sigma_y 1 \
            --seed "$seed_val" \
            --normal_output \
            --test_steps 100 \
            --pf_verification \
            --pf_N "$pf_n_val" \
            --sigma_reg None \
            --pf_save_figure
        
        echo "Done."
        echo ""
    done
done

# SEEDS=(0 1 2 3 4 5 6 7 8 9 42)
# PARTICLE_NUMBERS=(1000 2000 5000 10000 20000 50000 100000 200000 500000 1000000)
# dataset="lorenz96"

# for seed_val in "${SEEDS[@]}"; do
#     for pf_n_val in "${PARTICLE_NUMBERS[@]}"; do
        
#         echo "============================================================"
#         echo "Running with Seed: $seed_val and Particle Count (pf_N): $pf_n_val"
#         echo "============================================================"
        
#         python gen_pf_results.py \
#             --dataset $dataset \
#             --sigma_y 1 \
#             --seed "$seed_val" \
#             --normal_output \
#             --test_steps 500 \
#             --pf_verification \
#             --pf_N "$pf_n_val" \
#             --sigma_reg None
        
#         echo "Done."
#         echo ""
#     done
# done

echo "All experiments finished."
echo "Job completed on $(date)"