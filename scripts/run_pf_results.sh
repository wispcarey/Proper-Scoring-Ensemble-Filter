#!/bin/bash

set -e

cd ..

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    else
        echo "Error: neither 'python' nor 'python3' is available in PATH." >&2
        exit 127
    fi
fi

echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Python: $(command -v "$PYTHON_BIN")"
echo "----------------------------------------------------"

echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Python: $(command -v "$PYTHON_BIN")"
echo "----------------------------------------------------"

# SEEDS=(0 1 2 3 4 5 6 7 8 9 10 42)
SEEDS=(42)
# PARTICLE_NUMBERS=(500 1000 2000 5000 10000 20000 50000 100000 200000 500000 1000000)
PARTICLE_NUMBERS=(1000000)

for seed_val in "${SEEDS[@]}"; do
    for pf_n_val in "${PARTICLE_NUMBERS[@]}"; do
        
        echo "============================================================"
        echo "Running with Seed: $seed_val and Particle Count (pf_N): $pf_n_val"
        echo "============================================================"
        
        "$PYTHON_BIN" gen_pf_results.py \
            --dataset complex2d \
            --seed "$seed_val" \
            --normal_output \
            --test_steps 200 \
            --pf_verification \
            --pf_N "$pf_n_val" \
            --sigma_reg None \
            --pf_save_figure
        
        echo "Done."
        echo ""
    done
done

for seed_val in "${SEEDS[@]}"; do
    for pf_n_val in "${PARTICLE_NUMBERS[@]}"; do
        
        echo "============================================================"
        echo "Running with Seed: $seed_val and Particle Count (pf_N): $pf_n_val"
        echo "============================================================"
        
        "$PYTHON_BIN" gen_pf_results.py \
            --dataset doubling1d \
            --seed "$seed_val" \
            --normal_output \
            --test_steps 200 \
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
