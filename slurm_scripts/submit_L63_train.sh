#!/bin/bash

# Description: Batch submit Slurm jobs for train.py (via slurm_train.sh).
# Input: List of (DATASET, EPOCHS, N, SIGMA_Y, VERSION, LOSS_TYPE, LOSS_WEIGHTS, USE_PF, LR, SUFFIX).
# Output: Submits jobs via sbatch using exported environment variables.

# Target Slurm script
SLURM_SCRIPT="slurm_L63_train.sh"

# Default values
DEF_DATASET="lorenz63"
DEF_EPOCHS=1000
DEF_N=10
DEF_SIGMA_Y=1
DEF_VERSION="EtE-LRes"
DEF_LOSS="es"
DEF_WEIGHTS="none"
DEF_USE_PF="true"
DEF_LR="default"
DEF_SUFFIX=""

# List of specific experiments to run
# Format: "DATASET EPOCHS N SIGMA_Y VERSION LOSS_TYPE LOSS_WEIGHTS USE_PF LEARNING_RATE SUFFIX"
# Note: If a value is missing (end of string), it defaults to the DEF variables above.
# EXPERIMENTS=(
#     "lorenz63 500 10 1.0 EtE-LRes pre_nll None true 1e-3"
#     "lorenz63 500 10 1.0 CorrTerms pre_nll None true 1e-3"
#     "lorenz63 500 10 1.0 EtE-LRes nll None true 1e-3"
#     "lorenz63 500 10 1.0 CorrTerms nll None true 1e-3"
#     "lorenz63 500 10 1.0 EtE-LRes pre_nll,wpf_ed None true 1e-3"
#     "lorenz63 500 10 1.0 CorrTerms pre_nll,wpf_ed None true 1e-3"
#     "lorenz63 500 10 1.0 EtE-LRes pre_nll,wpf_ammd None true 1e-3"
#     "lorenz63 500 10 1.0 CorrTerms pre_nll,wpf_ammd None true 1e-3"
#     "lorenz63 500 10 1.0 EtE-LRes nll,wpf_ed None true 1e-3"
#     "lorenz63 500 10 1.0 CorrTerms nll,wpf_ed None true 1e-3"
#     "lorenz63 500 10 1.0 EtE-LRes nll,wpf_ammd None true 1e-3"
#     "lorenz63 500 10 1.0 CorrTerms nll,wpf_ammd None true 1e-3"
#     "lorenz63 500 10 1.0 EtE-LRes nl2 None true 1e-3 comp_ss"
#     "lorenz63 500 10 1.0 CorrTerms nl2 None true 1e-3 comp_ss"
#     "lorenz63 500 10 1.0 EtE-LRes es None true 1e-3 comp_ss"
#     "lorenz63 500 10 1.0 CorrTerms es None true 1e-3 comp_ss"
# )
EXPERIMENTS=(
    "lorenz63 500 10 1.0 EtE-LRes nl2 None true 1e-3 comp_ss"
    "lorenz63 500 10 1.0 CorrTerms nl2 None true 1e-3 comp_ss"
)

# Iterate and submit
for exp in "${EXPERIMENTS[@]}"; do
    # Parse the string into variables
    read -r dataset epochs n sigma_y version loss_type loss_weights use_pf learning_rate suffix <<< "$exp"

    # Export variables to current shell environment so --export=ALL can pick them up.
    export DATASET=${dataset:-$DEF_DATASET}
    export EPOCHS=${epochs:-$DEF_EPOCHS}
    export N=${n:-$DEF_N}
    export SIGMA_Y=${sigma_y:-$DEF_SIGMA_Y}
    export VERSION=${version:-$DEF_VERSION}
    export LOSS_TYPE=${loss_type:-$DEF_LOSS}
    export LOSS_WEIGHTS=${loss_weights:-$DEF_WEIGHTS}
    export USE_PF=${use_pf:-$DEF_USE_PF}
    export LEARNING_RATE=${learning_rate:-$DEF_LR}
    export SUFFIX=${suffix:-$DEF_SUFFIX}

    # --- Job Name Construction ---
    JOB_NAME="${DATASET}-${LOSS_TYPE}-N${N}"
    if [ -n "$SUFFIX" ]; then
        JOB_NAME="${JOB_NAME}-${SUFFIX}"
    fi

    echo "Submitting job: $JOB_NAME (Loss=$LOSS_TYPE, Weights=$LOSS_WEIGHTS, V=$VERSION, PF=$USE_PF)"

    # Submit using --export=ALL
    sbatch -J "$JOB_NAME" \
       --time=12:00:00 \
       --export=ALL \
       $SLURM_SCRIPT
done