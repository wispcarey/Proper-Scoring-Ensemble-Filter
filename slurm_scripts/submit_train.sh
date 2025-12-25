#!/bin/bash

# Description: Batch submit Slurm jobs for train.py (via slurm_train.sh).
# Input: List of (DATASET, EPOCHS, N, SIGMA_Y, VERSION, LOSS_TYPE, USE_PF).
# Output: Submits jobs via sbatch with specific environment variables.

# Target Slurm script
SLURM_SCRIPT="slurm_train.sh"

# Default values (used if a column is missing in the tuple string)
DEF_DATASET="lorenz63"
DEF_EPOCHS=1000
DEF_N=10
DEF_SIGMA_Y=1
DEF_VERSION="EtE-LRes"
DEF_LOSS="es"
DEF_USE_PF="true"

# List of specific experiments to run
# Format: "DATASET EPOCHS N SIGMA_Y VERSION LOSS_TYPE USE_PF"
# You can leave trailing parameters empty to use defaults.
EXPERIMENTS=(
    "lorenz63 1000 10 1 EtE-LRes es true"
    "lorenz63 1000 10 1 EtE-LRes nes true"
    "lorenz63 1000 10 1 EtE-LRes nl2 true"
    "lorenz63 1000 10 1 EtE-LRes l2 true"
)

# Iterate and submit
for exp in "${EXPERIMENTS[@]}"; do
    # Parse the string into variables
    read -r dataset epochs n sigma_y version loss_type use_pf <<< "$exp"

    # Apply defaults if variable is empty
    dataset=${dataset:-$DEF_DATASET}
    epochs=${epochs:-$DEF_EPOCHS}
    n=${n:-$DEF_N}
    sigma_y=${sigma_y:-$DEF_SIGMA_Y}
    version=${version:-$DEF_VERSION}
    loss_type=${loss_type:-$DEF_LOSS}
    use_pf=${use_pf:-$DEF_USE_PF}

    echo "Submitting job: D=$dataset, Ep=$epochs, N=$n, Sig=$sigma_y, V=$version, Loss=$loss_type, PF=$use_pf"

    # Submit using --export to pass variables to the Slurm script
    sbatch --export=ALL,DATASET=$dataset,EPOCHS=$epochs,N=$n,SIGMA_Y=$sigma_y,VERSION=$version,LOSS_TYPE=$loss_type,USE_PF=$use_pf $SLURM_SCRIPT
done