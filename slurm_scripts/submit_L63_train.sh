#!/bin/bash

# Description: Batch submit Slurm jobs for train.py (via slurm_train.sh).
# Input: List of (DATASET, EPOCHS, N, SIGMA_Y, VERSION, LOSS_TYPE, LOSS_WEIGHTS, USE_PF, LR, OBS_FN, WEIGHT_DECAY, ADAPTIVE_SIGMA_Y, SUFFIX).
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
DEF_OBS_FN="identity"
DEF_WEIGHT_DECAY="0"
DEF_SUFFIX=""

compute_time_limit() {
    local epochs="$1"
    local base_hours=12
    local max_hours=48
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

# List of specific experiments to run
# Format: "DATASET EPOCHS N SIGMA_Y VERSION LOSS_TYPE LOSS_WEIGHTS USE_PF LEARNING_RATE OBS_FN WEIGHT_DECAY ADAPTIVE_SIGMA_Y SUFFIX"
# Note: ADAPTIVE_SIGMA_Y is required and must be explicitly set to true/false.
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
    # "lorenz63 1000 10 2.0 EtE-LRes es None false 1e-3 square_root 0 true None"
    # "lorenz63 1000 10 2.0 EtE-LRes nl2 None false 1e-3 square_root 0 true None"
    # "lorenz63 1000 10 2.0 CorrTerms es None false 1e-3 square_root 0.01 true None"
    # "lorenz63 1000 10 2.0 CorrTerms nl2 None false 1e-3 square_root 0.01 true None"
    # "lorenz63 1000 10 2.0 EtE-LRes es None false 1e-3 arctan 0 true None"
    # "lorenz63 1000 10 2.0 EtE-LRes nl2 None false 1e-3 arctan 0 true None"
    # "lorenz63 1000 10 2.0 CorrTerms es None false 1e-3 arctan 0.01 true None"
    # "lorenz63 1000 10 2.0 CorrTerms nl2 None false 1e-3 arctan 0.01 true None"
    # "lorenz63 1000 10 2.0 EtE-LRes es None false 1e-3 square 0 true None"
    # "lorenz63 1000 10 2.0 EtE-LRes nl2 None false 1e-3 square 0 true None"
    # "lorenz63 1000 10 2.0 CorrTerms es None false 1e-3 square 0.01 true None"
    # "lorenz63 1000 10 2.0 CorrTerms nl2 None false 1e-3 square 0.01 true None"
    "lorenz63 1000 10 2.0 EtE-LRes es None false 1e-3 default 0 true None"
    "lorenz63 1000 10 2.0 EtE-LRes nl2 None false 1e-3 default 0 true None"
    "lorenz63 1000 10 2.0 CorrTerms es None false 1e-3 default 0.01 true None"
    "lorenz63 1000 10 2.0 CorrTerms nl2 None false 1e-3 default 0.01 true None"
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


# Iterate and submit
for exp in "${EXPERIMENTS[@]}"; do
    # Parse the string into variables
    read -r dataset epochs n sigma_y version loss_type loss_weights use_pf learning_rate obs_fn weight_decay adaptive_sigma_y suffix <<< "$exp"

    if ! validate_adaptive_sigma_y "$adaptive_sigma_y"; then
        echo "Malformed experiment entry: $exp" >&2
        exit 1
    fi

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
    export OBS_FN=${obs_fn:-$DEF_OBS_FN}
    export WEIGHT_DECAY=${weight_decay:-$DEF_WEIGHT_DECAY}
    export ADAPTIVE_SIGMA_Y=$adaptive_sigma_y
    export SUFFIX=${suffix:-$DEF_SUFFIX}
    TIME_LIMIT=$(compute_time_limit "$EPOCHS")

    # --- Job Name Construction ---
    JOB_NAME="${DATASET}-${LOSS_TYPE}-N${N}"
    if [ -n "$SUFFIX" ]; then
        JOB_NAME="${JOB_NAME}-${SUFFIX}"
    fi

    echo "Submitting job: $JOB_NAME (Loss=$LOSS_TYPE, Weights=$LOSS_WEIGHTS, V=$VERSION, PF=$USE_PF, LR=$LEARNING_RATE, ObsFn=$OBS_FN, WD=$WEIGHT_DECAY, Adaptive=$ADAPTIVE_SIGMA_Y, Time=$TIME_LIMIT)"

    # Submit using --export=ALL
    sbatch -J "$JOB_NAME" \
       --time="$TIME_LIMIT" \
       --export=ALL \
       $SLURM_SCRIPT
done
