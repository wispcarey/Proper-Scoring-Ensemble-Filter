#!/bin/bash

# Description: Batch submit Slurm jobs for train.py (via slurm_highdim_train.sh).
# Input: List of (DATASET, EPOCHS, N, SIGMA_Y, VERSION, LOSS_TYPE, SUFFIX).
# Output: Submits jobs via sbatch with specific environment variables.

# Target Slurm script
SLURM_SCRIPT="slurm_highdim_train.sh"

# Default values
DEF_DATASET="lorenz63"
DEF_EPOCHS=1000
DEF_N=10
DEF_SIGMA_Y=1
DEF_VERSION="EtE-LRes"
DEF_LOSS="es"
DEF_SUFFIX=""

# List of specific experiments to run
# Format: "DATASET EPOCHS N SIGMA_Y VERSION LOSS_TYPE SUFFIX"
# Note: Removed USE_PF and LEARNING_RATE columns.
EXPERIMENTS=(
    "lorenz96 1000 10 1 EtE-LRes es"
    "ks 1000 10 1 EtE-LRes es"
)

# Iterate and submit
for exp in "${EXPERIMENTS[@]}"; do
    # Parse the string into variables
    read -r dataset epochs n sigma_y version loss_type suffix <<< "$exp"

    # Apply defaults if variable is empty
    dataset=${dataset:-$DEF_DATASET}
    epochs=${epochs:-$DEF_EPOCHS}
    n=${n:-$DEF_N}
    sigma_y=${sigma_y:-$DEF_SIGMA_Y}
    version=${version:-$DEF_VERSION}
    loss_type=${loss_type:-$DEF_LOSS}
    suffix=${suffix:-$DEF_SUFFIX}

    # --- Job Name Construction ---
    # Example Name: lorenz63-es-N10-_1e-3
    JOB_NAME="${dataset}-${loss_type}-N${n}"
    if [ -n "$suffix" ]; then
        JOB_NAME="${JOB_NAME}${suffix}" 
    fi

    echo "Submitting job: $JOB_NAME (Sig=$sigma_y, V=$version, Suffix=$suffix)"

    # Submit using --export to pass variables
    sbatch -J "$JOB_NAME" \
           --export=ALL,DATASET=$dataset,EPOCHS=$epochs,N=$n,SIGMA_Y=$sigma_y,VERSION=$version,LOSS_TYPE=$loss_type,SUFFIX=$suffix \
           $SLURM_SCRIPT
done