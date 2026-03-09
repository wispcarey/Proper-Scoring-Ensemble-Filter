#!/bin/bash

# Description:
# Submit benchmark grid-search jobs. Each submitted Slurm job fixes
# one dataset/method/obs_fn trial and iterates only over ensemble size N.
#
# Usage:
# bash submit_grid_search_benchmark.sh
#
# Optional environment overrides:
# DEFAULT_TIME_LIMIT=2-00:00:00
# ENSEMBLE_SIZES="5 10 15 20 40 60 100"
# GRID_SEARCH_NUM_SEEDS=4
# SEED=42

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="$SCRIPT_DIR/slurm_grid_search_benchmark.sh"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/grid_search/out"
ERROR_DIR="$SCRIPT_DIR/grid_search/err"

DEFAULT_TIME_LIMIT="${DEFAULT_TIME_LIMIT:-2-00:00:00}"
ENSEMBLE_SIZES="${ENSEMBLE_SIZES:-5 10 15 20 40 60 100}"
GRID_SEARCH_NUM_SEEDS="${GRID_SEARCH_NUM_SEEDS:-4}"
SEED="${SEED:-42}"
CPU_WORKERS=32
CPUS_PER_TASK=32

mkdir -p "$OUTPUT_DIR" "$ERROR_DIR"

sanitize_token() {
    echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9._-]/-/g'
}

validate_positive_int() {
    local raw="$1"
    local name="$2"
    if [[ ! "$raw" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: $name must be a positive integer, got '$raw'." >&2
        exit 1
    fi
}

validate_positive_int "$GRID_SEARCH_NUM_SEEDS" "GRID_SEARCH_NUM_SEEDS"
validate_positive_int "$SEED" "SEED"

resolve_time_limit() {
    local dataset="$1"
    case "$dataset" in
        lorenz63) echo "1-00:00:00" ;;
        doubling1d) echo "0-18:00:00" ;;
        lorenz96) echo "2-00:00:00" ;;
        *) echo "$DEFAULT_TIME_LIMIT" ;;
    esac
}

# Format per trial:
# DATASET METHOD OBS_FN
TRIALS=(
    "lorenz96 EnKF square"
    "lorenz96 ESRF square"
    "lorenz96 LETKF square"
    "lorenz96 iEnKS-PertObs square"
    "lorenz96 EnKF arctan"
    "lorenz96 ESRF arctan"
    "lorenz96 LETKF arctan"
    "lorenz96 iEnKS-PertObs arctan"
    "lorenz96 EnKF identity"
    "lorenz96 ESRF identity"
    "lorenz96 LETKF identity"
    "lorenz96 iEnKS-PertObs identity"
    "lorenz63 EnKF square"
    "lorenz63 ESRF square"
    "lorenz63 iEnKS-PertObs square"
    "lorenz63 EnKF arctan"
    "lorenz63 ESRF arctan"
    "lorenz63 iEnKS-PertObs arctan"
    "lorenz63 EnKF identity"
    "lorenz63 ESRF identity"
    "lorenz63 iEnKS-PertObs identity"
    "doubling1d EnKF cos2pi"
    "doubling1d ESRF cos2pi"
    "doubling1d iEnKS-PertObs cos2pi"
)

for trial in "${TRIALS[@]}"; do
    read -r dataset method obs_fn <<< "$trial"

    if [ -z "$dataset" ] || [ -z "$method" ] || [ -z "$obs_fn" ]; then
        echo "Error: invalid trial entry '$trial'." >&2
        exit 1
    fi

    job_name="gs-$(sanitize_token "$dataset")-$(sanitize_token "$method")-$(sanitize_token "$obs_fn")"
    time_limit="$(resolve_time_limit "$dataset")"

    echo "Submitting: $job_name"
    echo "  DATASET=$dataset METHOD=$method OBS_FN=$obs_fn"
    echo "  REPO_ROOT=$REPO_ROOT"
    echo "  ENSEMBLE_SIZES=$ENSEMBLE_SIZES GRID_SEARCH_NUM_SEEDS=$GRID_SEARCH_NUM_SEEDS SEED=$SEED TIME_LIMIT=$time_limit CPU_WORKERS=$CPU_WORKERS"

    sbatch \
        -J "$job_name" \
        --time="$time_limit" \
        --cpus-per-task="$CPUS_PER_TASK" \
        --output="$OUTPUT_DIR/%x.%j.out" \
        --error="$ERROR_DIR/%x.%j.err" \
        --export=ALL,REPO_ROOT="$REPO_ROOT",DATASET="$dataset",METHOD="$method",OBS_FN="$obs_fn",ENSEMBLE_SIZES="$ENSEMBLE_SIZES",GRID_SEARCH_NUM_SEEDS="$GRID_SEARCH_NUM_SEEDS",SEED="$SEED",CPU_WORKERS="$CPU_WORKERS",ADAPTIVE_SIGMA_Y=true \
        "$SLURM_SCRIPT"
done
