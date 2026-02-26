#!/bin/bash

# Description: Batch submit Slurm jobs for low-dim train.py runs.
# Input: List of
# (EXP_SETTING EPOCHS N BATCH_SIZE SIGMA_Y VERSION LOSS_TYPE LOSS_WEIGHTS USE_PF
#  LEARNING_RATE OBS_FN WEIGHT_DECAY ADAPTIVE_SIGMA_Y SUFFIX)
# EXP_SETTING supports: 1d/2d (also backward-compatible with doubling1d/complex2d).

SLURM_SCRIPT="slurm_doubling1d_train.sh"

# Default values
DEF_EPOCHS=100
DEF_VERSION="EtE-LRes"
DEF_LOSS="es"
DEF_WEIGHTS="None"
DEF_SUFFIX=""

set_defaults_by_exp_setting() {
    local exp_setting="$1"

    case "$exp_setting" in
        "1d"|"doubling1d")
            DEF_EXP_SETTING="1d"
            DEF_DATASET="doubling1d"
            DEF_N=30
            DEF_BATCH_SIZE=512
            DEF_SIGMA_Y=0.1
            DEF_USE_PF="true"
            DEF_LR="default"
            DEF_OBS_FN="default"
            DEF_WEIGHT_DECAY="0"
            ;;
        "2d"|"complex2d")
            DEF_EXP_SETTING="2d"
            DEF_DATASET="complex2d"
            DEF_N=30
            DEF_BATCH_SIZE=512
            DEF_SIGMA_Y=0.1
            DEF_USE_PF="true"
            DEF_LR="default"
            DEF_OBS_FN="default"
            DEF_WEIGHT_DECAY="0"
            ;;
        *)
            echo "Error: unsupported EXP_SETTING '$exp_setting'. Use 1d or 2d." >&2
            return 1
            ;;
    esac
}

compute_time_limit() {
    local epochs="$1"
    local base_hours=6
    local max_hours=24
    local hours

    if [ "$epochs" -le 100 ]; then
        hours=$base_hours
    else
        # Scale from 100-epoch baseline and round up.
        hours=$(( (epochs * base_hours + 100 - 1) / 100 ))
    fi

    if [ "$hours" -gt "$max_hours" ]; then
        hours=$max_hours
    fi

    printf "%02d:00:00" "$hours"
}

# Experiment list
# Format:
# "EXP_SETTING EPOCHS N BATCH_SIZE SIGMA_Y VERSION LOSS_TYPE LOSS_WEIGHTS USE_PF
#  LEARNING_RATE OBS_FN WEIGHT_DECAY ADAPTIVE_SIGMA_Y SUFFIX"
EXPERIMENTS=(
    "1d 500 30 512 0.2 EtE-LRes es None true default default 0 false None"
    "1d 500 30 512 0.2 EtE-LRes nl2 None true default default 0 false None"
    "1d 500 30 512 0.2 CorrTerms es None true default default 0 false None"
    "1d 500 30 512 0.2 CorrTerms nl2 None true default default 0 false None"
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
    read -r exp_setting epochs n batch_size sigma_y version loss_type loss_weights use_pf \
        learning_rate obs_fn weight_decay adaptive_sigma_y suffix <<< "$exp"

    if ! validate_adaptive_sigma_y "$adaptive_sigma_y"; then
        echo "Malformed experiment entry: $exp" >&2
        exit 1
    fi

    if ! set_defaults_by_exp_setting "$exp_setting"; then
        exit 1
    fi

    export EXP_SETTING=$DEF_EXP_SETTING
    export DATASET=$DEF_DATASET
    export EPOCHS=${epochs:-$DEF_EPOCHS}
    export N=${n:-$DEF_N}
    export BATCH_SIZE=${batch_size:-$DEF_BATCH_SIZE}
    export SIGMA_Y=${sigma_y:-$DEF_SIGMA_Y}
    export VERSION=${version:-$DEF_VERSION}
    export LOSS_TYPE=${loss_type:-$DEF_LOSS}
    export LOSS_WEIGHTS=${loss_weights:-$DEF_WEIGHTS}
    export USE_PF=${use_pf:-$DEF_USE_PF}
    export LEARNING_RATE=${learning_rate:-$DEF_LR}
    export OBS_FN=${obs_fn:-$DEF_OBS_FN}
    export WEIGHT_DECAY=${weight_decay:-$DEF_WEIGHT_DECAY}
    export ADAPTIVE_SIGMA_Y=$adaptive_sigma_y
    export SUFFIX=${suffix:-$DEF_SUFFIX}

    TIME_LIMIT=$(compute_time_limit "$EPOCHS")

    JOB_NAME="${EXP_SETTING}-${LOSS_TYPE}-N${N}-sy${SIGMA_Y}"
    if [ -n "$SUFFIX" ]; then
        JOB_NAME="${JOB_NAME}-${SUFFIX}"
    fi

    echo "Submitting job: $JOB_NAME (DATASET=$DATASET, EPOCHS=$EPOCHS, SIGMA_Y=$SIGMA_Y, BATCH=$BATCH_SIZE, PF=$USE_PF, LR=$LEARNING_RATE, Adaptive=$ADAPTIVE_SIGMA_Y, Time=$TIME_LIMIT)"

    sbatch -J "$JOB_NAME" \
        --time="$TIME_LIMIT" \
        --export=ALL \
        "$SLURM_SCRIPT"
done
