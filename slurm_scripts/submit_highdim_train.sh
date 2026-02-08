#!/bin/bash

# Description: Batch submit Slurm jobs for train.py
# Input: List of (DATASET, EPOCHS, N, SIGMA_Y, VERSION, LOSS_TYPE, LOSS_WEIGHTS, OBS_FN, WEIGHT_DECAY, SUFFIX)

SLURM_SCRIPT="slurm_highdim_train.sh"

# Default values
DEF_DATASET="lorenz96"
DEF_EPOCHS=1000
DEF_N=10
DEF_SIGMA_Y=1
DEF_VERSION="EtE-LRes"
DEF_LOSS="es"
DEF_WEIGHTS="none"
DEF_OBS_FN="identity"
DEF_WEIGHT_DECAY="0"
DEF_SUFFIX=""

# Experiment list
# EXPERIMENTS=(
#     #no nll
#     "lorenz96 500 10 1.0 EtE-LRes wpf_ed None identity 0"
#     "lorenz96 500 10 1.0 EtE-LRes wpf_st_ed None identity 0"
#     "lorenz96 500 10 1.0 CorrTerms wpf_ed None identity 0"
#     "lorenz96 500 10 1.0 CorrTerms wpf_st_ed None identity 0"
#     "lorenz96 500 10 1.0 EtE-LRes wpf_ammd None identity 0"
#     "lorenz96 500 10 1.0 EtE-LRes wpf_st_ammd None identity 0"
#     "lorenz96 500 10 1.0 CorrTerms wpf_ammd None identity 0"
#     "lorenz96 500 10 1.0 CorrTerms wpf_st_ammd None identity 0"
#     #no wpf
#     "lorenz96 500 10 1.0 EtE-LRes nll None identity 0"
#     "lorenz96 500 10 1.0 CorrTerms nll None identity 0"
#     "lorenz96 500 10 1.0 EtE-LRes nnll None identity 0"
#     "lorenz96 500 10 1.0 CorrTerms nnll None identity 0"
#     "lorenz96 500 10 1.0 EtE-LRes pre_nll None identity 0"
#     "lorenz96 500 10 1.0 CorrTerms pre_nll None identity 0"
#     "lorenz96 500 10 1.0 EtE-LRes pre_nnll None identity 0"
#     "lorenz96 500 10 1.0 CorrTerms pre_nnll None identity 0"
#     #pre_nll
#     "lorenz96 500 10 1.0 EtE-LRes pre_nll,wpf_ed None identity 0"
#     "lorenz96 500 10 1.0 EtE-LRes pre_nll,wpf_st_ed None identity 0"
#     "lorenz96 500 10 1.0 CorrTerms pre_nll,wpf_ed None identity 0"
#     "lorenz96 500 10 1.0 CorrTerms pre_nll,wpf_st_ed None identity 0"
#     "lorenz96 500 10 1.0 EtE-LRes pre_nll,wpf_ammd None identity 0"
#     "lorenz96 500 10 1.0 EtE-LRes pre_nll,wpf_st_ammd None identity 0"
#     "lorenz96 500 10 1.0 CorrTerms pre_nll,wpf_ammd None identity 0"
#     "lorenz96 500 10 1.0 CorrTerms pre_nll,wpf_st_ammd None identity 0"
#     #pre_nnll
#     "lorenz96 500 10 1.0 EtE-LRes pre_nnll,wpf_ed None identity 0"
#     "lorenz96 500 10 1.0 EtE-LRes pre_nnll,wpf_st_ed None identity 0"
#     "lorenz96 500 10 1.0 CorrTerms pre_nnll,wpf_ed None identity 0"
#     "lorenz96 500 10 1.0 CorrTerms pre_nnll,wpf_st_ed None identity 0"
#     "lorenz96 500 10 1.0 EtE-LRes pre_nnll,wpf_ammd None identity 0"
#     "lorenz96 500 10 1.0 EtE-LRes pre_nnll,wpf_st_ammd None identity 0"
#     "lorenz96 500 10 1.0 CorrTerms pre_nnll,wpf_ammd None identity 0"
#     "lorenz96 500 10 1.0 CorrTerms pre_nnll,wpf_st_ammd None identity 0"
#     #nll
#     "lorenz96 500 10 1.0 EtE-LRes nll,wpf_ed None identity 0"
#     "lorenz96 500 10 1.0 EtE-LRes nll,wpf_st_ed None identity 0"
#     "lorenz96 500 10 1.0 CorrTerms nll,wpf_ed None identity 0"
#     "lorenz96 500 10 1.0 CorrTerms nll,wpf_st_ed None identity 0"
#     "lorenz96 500 10 1.0 EtE-LRes nll,wpf_ammd None identity 0"
#     "lorenz96 500 10 1.0 EtE-LRes nll,wpf_st_ammd None identity 0"
#     "lorenz96 500 10 1.0 CorrTerms nll,wpf_ammd None identity 0"
#     "lorenz96 500 10 1.0 CorrTerms nll,wpf_st_ammd None identity 0"
#     #nnll
#     "lorenz96 500 10 1.0 EtE-LRes nnll,wpf_ed None identity 0"
#     "lorenz96 500 10 1.0 EtE-LRes nnll,wpf_st_ed None identity 0"
#     "lorenz96 500 10 1.0 CorrTerms nnll,wpf_ed None identity 0"
#     "lorenz96 500 10 1.0 CorrTerms nnll,wpf_st_ed None identity 0"
#     "lorenz96 500 10 1.0 EtE-LRes nnll,wpf_ammd None identity 0"
#     "lorenz96 500 10 1.0 EtE-LRes nnll,wpf_st_ammd None identity 0"
#     "lorenz96 500 10 1.0 CorrTerms nnll,wpf_ammd None identity 0"
#     "lorenz96 500 10 1.0 CorrTerms nnll,wpf_st_ammd None identity 0"
# )

EXPERIMENTS=(
    "lorenz96 1000 10 1.0 EtE-LRes es None square_root 0"
    "lorenz96 1000 10 1.0 EtE-LRes nl2 None square_root 0"
    "lorenz96 1000 10 1.0 CorrTerms es None square_root 0.01"
    "lorenz96 1000 10 1.0 CorrTerms nl2 None square_root 0.01"
    "lorenz96 1000 10 1.0 EtE-LRes es None arctan 0"
    "lorenz96 1000 10 1.0 EtE-LRes nl2 None arctan 0"
    "lorenz96 1000 10 1.0 CorrTerms es None arctan 0.01"
    "lorenz96 1000 10 1.0 CorrTerms nl2 None arctan 0.01"
)

for exp in "${EXPERIMENTS[@]}"; do
    # Parse the string into variables
    read -r dataset epochs n sigma_y version loss_type loss_weights obs_fn weight_decay suffix <<< "$exp"

    # Export variables to environment (prevents comma-parsing issues in --export)
    export DATASET=${dataset:-$DEF_DATASET}
    export EPOCHS=${epochs:-$DEF_EPOCHS}
    export N=${n:-$DEF_N}
    export SIGMA_Y=${sigma_y:-$DEF_SIGMA_Y}
    export VERSION=${version:-$DEF_VERSION}
    export LOSS_TYPE=${loss_type:-$DEF_LOSS}
    export LOSS_WEIGHTS=${loss_weights:-$DEF_WEIGHTS}
    export OBS_FN=${obs_fn:-$DEF_OBS_FN}
    export WEIGHT_DECAY=${weight_decay:-$DEF_WEIGHT_DECAY}
    export SUFFIX=${suffix:-$DEF_SUFFIX}

    # Construct Job Name
    JOB_NAME="${DATASET}-${LOSS_TYPE}-N${N}"
    if [ "$SUFFIX" != "None" ] && [ -n "$SUFFIX" ]; then
        JOB_NAME="${JOB_NAME}-${SUFFIX}" 
    fi

    echo "Submitting job: $JOB_NAME (Loss=$LOSS_TYPE, Weights=$LOSS_WEIGHTS, ObsFn=$OBS_FN, WD=$WEIGHT_DECAY)"

    # Submit using --export=ALL to pass the exported environment variables
    sbatch -J "$JOB_NAME" \
           --time=12:00:00 \
           --export=ALL \
           "$SLURM_SCRIPT"
done
