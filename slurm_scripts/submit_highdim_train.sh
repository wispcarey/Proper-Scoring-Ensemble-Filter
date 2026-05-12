#!/bin/bash

# Description: Batch submit Slurm jobs for train.py
# Input: List of (DATASET, EPOCHS, N, SIGMA_Y, VERSION, LOSS_TYPE, LOSS_WEIGHTS, LEARNING_RATE, OBS_FN, WEIGHT_DECAY, ADAPTIVE_SIGMA_Y, SUFFIX)

SLURM_SCRIPT="slurm_highdim_train.sh"
GPU_TYPE="${1:-p100}"

# Default values
DEF_DATASET="lorenz96"
DEF_EPOCHS=1000
DEF_N=10
DEF_SIGMA_Y=1
DEF_VERSION="EtE-LRes"
DEF_LOSS="es"
DEF_WEIGHTS="none"
DEF_LR="default"
DEF_OBS_FN="identity"
DEF_WEIGHT_DECAY="0"
DEF_SUFFIX=""

compute_time_limit() {
    local epochs="$1"
    local base_hours=18
    local max_hours=72
    local hours
    local days
    local rem

    if [ "$epochs" -le 500 ]; then
        hours=$base_hours
    else
        # Scale from the 500-epoch baseline and round up to avoid under-requesting time.
        hours=$(( (epochs * base_hours + 500 - 1) / 500 ))
    fi

    if [ "$hours" -gt "$max_hours" ]; then
        hours=$max_hours
    fi

    if [ "$hours" -ge 24 ]; then
        days=$(( hours / 24 ))
        rem=$(( hours % 24 ))
        printf "%d-%02d:00:00" "$days" "$rem"
    else
        printf "%02d:00:00" "$hours"
    fi
}

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
    # "lorenz96 1000 10 1.0 EtE-LRes es None default square_root 0 true None"
    # "lorenz96 1000 10 1.0 EtE-LRes nl2 None default square_root 0 true None"
    # "lorenz96 1000 10 1.0 CorrTerms es None default square_root 0.01 true None"
    # "lorenz96 1000 10 1.0 CorrTerms nl2 None default square_root 0.01 true None"
    # "lorenz96 1000 10 1.0 EtE-LRes es None default arctan 0 true None"
    # "lorenz96 1000 10 1.0 EtE-LRes nl2 None default arctan 0 true None"
    # "lorenz96 1000 10 1.0 CorrTerms es None default arctan 0.01 true None"
    # "lorenz96 1000 10 1.0 CorrTerms nl2 None default arctan 0.01 true None"
    # "lorenz96 1000 10 1.0 EtE-LRes es None default square 0 true None"
    # "lorenz96 1000 10 1.0 EtE-LRes nl2 None default square 0 true None"
    # "lorenz96 1000 10 1.0 CorrTerms es None default square 0.01 true None"
    # "lorenz96 1000 10 1.0 CorrTerms nl2 None default square 0.01 true None"
    "lorenz96 1000 10 1.0 EtE-LRes es None default default 0 true None"
    # "lorenz96 1000 10 1.0 EtE-LRes nl2 None default default 0 true None"
    # "lorenz96 1000 10 1.0 CorrTerms es None default default 0.01 true None"
    # "lorenz96 1000 10 1.0 CorrTerms nl2 None default default 0.01 true None"
)

validate_adaptive_sigma_y() {
    local v="$1"
    case "$v" in
        true|false) return 0 ;;
        *)
            echo "Error: ADAPTIVE_SIGMA_Y must be 'true' or 'false'. Got '$v'" >&2
            return 1
            ;;
    esac
}

for exp in "${EXPERIMENTS[@]}"; do
    # Parse the string into variables
    read -r dataset epochs n sigma_y version loss_type loss_weights learning_rate obs_fn weight_decay adaptive_sigma_y suffix <<< "$exp"

    if ! validate_adaptive_sigma_y "$adaptive_sigma_y"; then
        echo "Malformed experiment entry: $exp" >&2
        exit 1
    fi

    # Export variables to environment (prevents comma-parsing issues in --export)
    export DATASET=${dataset:-$DEF_DATASET}
    export EPOCHS=${epochs:-$DEF_EPOCHS}
    export N=${n:-$DEF_N}
    export SIGMA_Y=${sigma_y:-$DEF_SIGMA_Y}
    export VERSION=${version:-$DEF_VERSION}
    export LOSS_TYPE=${loss_type:-$DEF_LOSS}
    export LOSS_WEIGHTS=${loss_weights:-$DEF_WEIGHTS}
    export LEARNING_RATE=${learning_rate:-$DEF_LR}
    export OBS_FN=${obs_fn:-$DEF_OBS_FN}
    export WEIGHT_DECAY=${weight_decay:-$DEF_WEIGHT_DECAY}
    export ADAPTIVE_SIGMA_Y=$adaptive_sigma_y
    export SUFFIX=${suffix:-$DEF_SUFFIX}
    TIME_LIMIT=$(compute_time_limit "$EPOCHS")

    # Construct Job Name
    JOB_NAME="${DATASET}-${LOSS_TYPE}-N${N}"
    if [ "$SUFFIX" != "None" ] && [ -n "$SUFFIX" ]; then
        JOB_NAME="${JOB_NAME}-${SUFFIX}" 
    fi

    echo "Submitting job: $JOB_NAME (Loss=$LOSS_TYPE, Weights=$LOSS_WEIGHTS, LR=$LEARNING_RATE, ObsFn=$OBS_FN, WD=$WEIGHT_DECAY, Adaptive=$ADAPTIVE_SIGMA_Y, Time=$TIME_LIMIT)"

    # Submit using --export=ALL to pass the exported environment variables
    sbatch -J "$JOB_NAME" \
           --time="$TIME_LIMIT" \
           --gres="gpu:${GPU_TYPE}:1" \
           --exclude="hpc-93-36" \
           --export=ALL \
           "$SLURM_SCRIPT"
done
