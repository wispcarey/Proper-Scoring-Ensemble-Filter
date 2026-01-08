#!/bin/bash

# Description: Batch submit Slurm jobs for train.py (via slurm_train.sh).
# Input: List of (DATASET, EPOCHS, N, SIGMA_Y, VERSION, LOSS_TYPE, USE_PF, LR, SUFFIX).
# Output: Submits jobs via sbatch with specific environment variables.

# Target Slurm script
SLURM_SCRIPT="slurm_L63_train.sh"

# Default values
DEF_DATASET="lorenz63"
DEF_EPOCHS=1000
DEF_N=10
DEF_SIGMA_Y=1
DEF_VERSION="EtE-LRes"
DEF_LOSS="es"
DEF_USE_PF="true"
DEF_LR="default"
DEF_SUFFIX=""

# List of specific experiments to run
# Format: "DATASET EPOCHS N SIGMA_Y VERSION LOSS_TYPE USE_PF LEARNING_RATE SUFFIX"
# Note: If a value is missing (end of string), it defaults to the DEF variables above.
EXPERIMENTS=(
    "lorenz63 1000 10 1 EtE-LRes es true 1e-3 _1e-3"
    "lorenz63 1000 10 1 EtE-LRes nl2 true 1e-3 _1e-3"
    "lorenz63 1000 10 1 EtE-LRes l2 true 1e-3 _1e-3"
    "lorenz96 1000 10 1 EtE-LRes es False 1e-3 _1e-3"
    "ks 1000 10 1 EtE-LRes es False 5e-4 _5e-4"
)

# Iterate and submit
for exp in "${EXPERIMENTS[@]}"; do
    # Parse the string into variables
    read -r dataset epochs n sigma_y version loss_type use_pf learning_rate suffix <<< "$exp"

    # Apply defaults if variable is empty
    dataset=${dataset:-$DEF_DATASET}
    epochs=${epochs:-$DEF_EPOCHS}
    n=${n:-$DEF_N}
    sigma_y=${sigma_y:-$DEF_SIGMA_Y}
    version=${version:-$DEF_VERSION}
    loss_type=${loss_type:-$DEF_LOSS}
    use_pf=${use_pf:-$DEF_USE_PF}
    learning_rate=${learning_rate:-$DEF_LR}
    suffix=${suffix:-$DEF_SUFFIX}

    # --- Job Name Construction ---
    # Example Name: lorenz63-es-N10-exp1
    JOB_NAME="${dataset}-${loss_type}-N${n}"
    if [ -n "$suffix" ]; then
        JOB_NAME="${JOB_NAME}-${suffix}"
    fi

    echo "Submitting job: $JOB_NAME (Sig=$sigma_y, V=$version, PF=$use_pf, LR=$learning_rate, Suffix=$suffix)"

    # Submit using --export to pass variables
    sbatch -J "$JOB_NAME" \
           --export=ALL,DATASET=$dataset,EPOCHS=$epochs,N=$n,SIGMA_Y=$sigma_y,VERSION=$version,LOSS_TYPE=$loss_type,USE_PF=$use_pf,LEARNING_RATE=$learning_rate,SUFFIX=$suffix \
           $SLURM_SCRIPT
done