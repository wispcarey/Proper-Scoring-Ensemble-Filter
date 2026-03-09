#!/bin/bash

# Description:
# Submit benchmark evaluation jobs based on scripts/run_benchmark_test.sh.
# Each Slurm job fixes one dataset/obs_fn/method-group and iterates over ensemble size N.
#
# Grouping rule:
# - EnKF and ESRF are submitted together in one job.
# - LETKF is submitted separately.
# - Each iEnKS variant is submitted separately.
# - Each obs_fn is submitted separately.
#
# Usage:
# bash submit_benchmark_test.sh
# bash submit_benchmark_test.sh a100

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="$SCRIPT_DIR/slurm_benchmark_test.sh"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/benchmark_test/out"
ERROR_DIR="$SCRIPT_DIR/benchmark_test/err"
GPU_TYPE="${1:-p100}"

DEFAULT_TIME_LIMIT="${DEFAULT_TIME_LIMIT:-06:00:00}"
ENSEMBLE_SIZES="${ENSEMBLE_SIZES:-10 15 20 40 60 100}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
NUM_LOADER_WORKERS="${NUM_LOADER_WORKERS:-8}"
NORMAL_OUTPUT="${NORMAL_OUTPUT:-true}"
SAVE_TEST_FIGURES="${SAVE_TEST_FIGURES:-true}"
PF_N="${PF_N:-1000000}"
PYTHON_BIN="${PYTHON_BIN:-/home/bhchen/miniconda3/bin/python}"

mkdir -p "$OUTPUT_DIR" "$ERROR_DIR"

sanitize_token() {
    echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9._-]/-/g'
}

validate_bool() {
    local v
    v="$(echo "${1:-}" | tr '[:upper:]' '[:lower:]')"
    case "$v" in
        true|false|1|0|yes|no|y|n|on|off) return 0 ;;
        *)
            echo "Error: boolean value is invalid: '$1'." >&2
            return 1
            ;;
    esac
}

validate_positive_int() {
    local raw="$1"
    local name="$2"
    if [[ ! "$raw" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: $name must be a positive integer, got '$raw'." >&2
        exit 1
    fi
}

method_group_label() {
    local methods="$1"
    methods="${methods//,/ }"
    methods="${methods//+/ }"
    echo "$methods" | tr '[:upper:]' '[:lower:]' | xargs | tr ' ' '-'
}

validate_positive_int "$SEED" "SEED"
validate_positive_int "$PF_N" "PF_N"
validate_positive_int "$NUM_LOADER_WORKERS" "NUM_LOADER_WORKERS"
validate_bool "$NORMAL_OUTPUT"
validate_bool "$SAVE_TEST_FIGURES"
case "${DEVICE,,}" in
    cuda|cpu) ;;
    *)
        echo "Error: DEVICE must be 'cuda' or 'cpu', got '$DEVICE'." >&2
        exit 1
        ;;
esac

# Format per entry:
# DATASET SIGMA_Y OBS_FN METHODS USE_LOCALIZATION PF_VERIFICATION ADAPTIVE_SIGMA_Y
# METHODS uses '+' inside one field, so one job can run a small method group.
EXPERIMENTS=(
    "lorenz63 2.0 identity EnKF+ESRF false true true"
    "lorenz63 2.0 identity iEnKS-PertObs false true true"
    "lorenz63 2.0 identity iEnKS-Sqrt false true true"
    "lorenz63 2.0 identity iEnKS-Order1 false true true"
    "lorenz63 2.0 arctan EnKF+ESRF false true true"
    "lorenz63 2.0 arctan iEnKS-PertObs false true true"
    "lorenz63 2.0 arctan iEnKS-Sqrt false true true"
    "lorenz63 2.0 arctan iEnKS-Order1 false true true"
    "lorenz63 2.0 square EnKF+ESRF false true true"
    "lorenz63 2.0 square iEnKS-PertObs false true true"
    "lorenz63 2.0 square iEnKS-Sqrt false true true"
    "lorenz63 2.0 square iEnKS-Order1 false true true"
)

for exp in "${EXPERIMENTS[@]}"; do
    read -r dataset sigma_y obs_fn methods use_localization pf_verification adaptive_sigma_y <<< "$exp"

    validate_bool "$use_localization"
    validate_bool "$pf_verification"
    validate_bool "$adaptive_sigma_y"

    group_label="$(method_group_label "$methods")"
    job_name="bench-$(sanitize_token "$dataset")-$(sanitize_token "$obs_fn")-$(sanitize_token "$group_label")"

    echo "Submitting: $job_name"
    echo "  DATASET=$dataset SIGMA_Y=$sigma_y OBS_FN=$obs_fn METHODS=$methods"
    echo "  USE_LOCALIZATION=$use_localization PF_VERIFICATION=$pf_verification ADAPTIVE_SIGMA_Y=$adaptive_sigma_y"
    echo "  DEVICE=$DEVICE ENSEMBLE_SIZES=$ENSEMBLE_SIZES PF_N=$PF_N"

    sbatch_args=(
        -J "$job_name"
        --time="$DEFAULT_TIME_LIMIT"
        --output="$OUTPUT_DIR/%x.%j.out"
        --error="$ERROR_DIR/%x.%j.err"
    )
    if [ "${DEVICE,,}" = "cuda" ]; then
        sbatch_args+=(--gres="gpu:${GPU_TYPE}:1")
    fi

    sbatch \
        "${sbatch_args[@]}" \
        --export=ALL,REPO_ROOT="$REPO_ROOT",PYTHON_BIN="$PYTHON_BIN",DATASET="$dataset",SIGMA_Y="$sigma_y",OBS_FN="$obs_fn",METHODS="$methods",ENSEMBLE_SIZES="$ENSEMBLE_SIZES",SEED="$SEED",DEVICE="$DEVICE",NUM_LOADER_WORKERS="$NUM_LOADER_WORKERS",ADAPTIVE_SIGMA_Y="$adaptive_sigma_y",USE_LOCALIZATION="$use_localization",NORMAL_OUTPUT="$NORMAL_OUTPUT",SAVE_TEST_FIGURES="$SAVE_TEST_FIGURES",PF_VERIFICATION="$pf_verification",PF_N="$PF_N" \
        "$SLURM_SCRIPT"
done
