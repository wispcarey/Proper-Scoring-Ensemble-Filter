#!/bin/bash

# Description: Batch submit Slurm jobs for specific hyperparameter quintuplets.
# Input: Hardcoded list of (EPOCHS, LOSS_TYPE, LR, DIM, HIDDEN_DIM, SUFFIX).
# Output: Submits jobs via sbatch with environment variables.

# Target Slurm script
SLURM_SCRIPT="slurm_linear_toy.sh"

# Default values
DEFAULT_HIDDEN_DIM=16
DEFAULT_SUFFIX="default"

# List of specific experiments to run
# Format: "EPOCHS LOSS_TYPE LEARNING_RATE DIM HIDDEN_DIM SUFFIX"
# Note: To use a SUFFIX, HIDDEN_DIM must be specified (cannot be omitted).
EXPERIMENTS=(
    "300 nes 1e-2 20 16 _20_1e-3"
    "300 es 1e-2 20 16 _20_1e-3"
    "300 nl2 1e-2 20 16 _20_1e-3"
    "300 l2 1e-2 20 16 _20_1e-3"
    "300 nes 5e-3 20 16 _20_5e-4"
    "300 es 5e-3 20 16 _20_5e-4"
    "300 nl2 5e-3 20 16 _20_5e-4"
    "300 l2 5e-3 20 16 _20_5e-4"
)

# Iterate through the list and submit jobs
for exp in "${EXPERIMENTS[@]}"; do
    # Parse the string into variables
    read -r epochs loss lr dim hidden_dim suffix <<< "$exp"

    # Apply defaults
    if [[ -z "$hidden_dim" ]]; then hidden_dim=$DEFAULT_HIDDEN_DIM; fi
    if [[ -z "$suffix" ]]; then suffix=$DEFAULT_SUFFIX; fi

    echo "Submitting job: Loss=$loss, Epochs=$epochs, LR=$lr, Dim=$dim, Hidden=$hidden_dim, Suffix=$suffix"

    # Submit using --export to pass variables
    sbatch --export=ALL,EPOCHS=$epochs,LOSS_TYPE=$loss,LR=$lr,DIM=$dim,HIDDEN_DIM=$hidden_dim,SUFFIX=$suffix $SLURM_SCRIPT
done