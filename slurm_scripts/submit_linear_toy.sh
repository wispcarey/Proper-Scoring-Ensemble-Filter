#!/bin/bash

# Description: Batch submit Slurm jobs for specific hyperparameter quintuplets.
# Input: Hardcoded list of (EPOCHS, LOSS_TYPE, LR, DIM, HIDDEN_DIM).
# Output: Submits jobs via sbatch with environment variables.

# Target Slurm script
SLURM_SCRIPT="slurm_linear_toy.sh"

# Default hidden_dim if not provided in an experiment line
DEFAULT_HIDDEN_DIM=16

# List of specific experiments to run
# Format: "EPOCHS LOSS_TYPE LEARNING_RATE DIM [HIDDEN_DIM]"
# If HIDDEN_DIM is omitted, DEFAULT_HIDDEN_DIM will be used.
EXPERIMENTS=(
    "300 nes 1e-2 20 16"
    "300 es 1e-2 20 16"
    "300 nl2 1e-2 20 16"
    "300 l2 1e-2 20 16"
    "300 nes 5e-3 20 16"
    "300 es 5e-3 20 16"
    "300 nl2 5e-3 20 16"
    "300 l2 5e-3 20 16"
)

# Iterate through the list and submit jobs
for exp in "${EXPERIMENTS[@]}"; do
    # Parse the string into variables; hidden_dim may be empty
    read -r epochs loss lr dim hidden_dim <<< "$exp"

    # Apply default if hidden_dim not provided
    if [[ -z "$hidden_dim" ]]; then
        hidden_dim=$DEFAULT_HIDDEN_DIM
    fi

    echo "Submitting job: Loss=$loss, Epochs=$epochs, LR=$lr, Dim=$dim, HiddenDim=$hidden_dim"

    # Submit using --export to pass variables (including DIM and HIDDEN_DIM) to the Slurm script
    sbatch --export=ALL,EPOCHS=$epochs,LOSS_TYPE=$loss,LR=$lr,DIM=$dim,HIDDEN_DIM=$hidden_dim $SLURM_SCRIPT
done
