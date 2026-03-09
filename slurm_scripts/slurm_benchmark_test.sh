#!/bin/bash

# Submit this script with e.g.:
# sbatch --export=ALL,REPO_ROOT=/path/to/repo,DATASET=lorenz63,SIGMA_Y=2.0,OBS_FN=identity,METHODS=EnKF+ESRF slurm_benchmark_test.sh

#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -J "bench-test"
#SBATCH --mail-user=bhchen@caltech.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-}"
if [ -z "$REPO_ROOT" ]; then
    echo "Error: REPO_ROOT is not set. Submit via submit_benchmark_test.sh or export REPO_ROOT explicitly." >&2
    exit 1
fi
if [ ! -f "$REPO_ROOT/evaluate_benchmark.py" ]; then
    echo "Error: REPO_ROOT='$REPO_ROOT' does not contain evaluate_benchmark.py." >&2
    exit 1
fi
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/bhchen/miniconda3/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    echo "Error: required Python interpreter is not executable: $PYTHON_BIN" >&2
    exit 127
fi

parse_bool() {
    local raw="${1:-false}"
    local v
    v="$(echo "$raw" | tr '[:upper:]' '[:lower:]')"
    case "$v" in
        true|1|yes|y|on) echo "true" ;;
        false|0|no|n|off|"") echo "false" ;;
        *)
            echo "Error: invalid boolean value '$raw'." >&2
            exit 1
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

DATASET="${DATASET:-lorenz63}"
SIGMA_Y="${SIGMA_Y:-2.0}"
OBS_FN="${OBS_FN:-identity}"
METHODS_RAW="${METHODS:-EnKF+ESRF}"
SEED="${SEED:-42}"
ENSEMBLE_SIZES_RAW="${ENSEMBLE_SIZES:-10 15 20 40 60 100}"
DEVICE="${DEVICE:-cuda}"
NUM_LOADER_WORKERS="${NUM_LOADER_WORKERS:-8}"
ADAPTIVE_SIGMA_Y="$(parse_bool "${ADAPTIVE_SIGMA_Y:-true}")"
USE_LOCALIZATION="$(parse_bool "${USE_LOCALIZATION:-false}")"
NORMAL_OUTPUT="$(parse_bool "${NORMAL_OUTPUT:-true}")"
SAVE_TEST_FIGURES="$(parse_bool "${SAVE_TEST_FIGURES:-true}")"
PF_VERIFICATION="$(parse_bool "${PF_VERIFICATION:-true}")"
PF_N="${PF_N:-1000000}"

validate_positive_int "$SEED" "SEED"
validate_positive_int "$NUM_LOADER_WORKERS" "NUM_LOADER_WORKERS"
if [ "$PF_VERIFICATION" = "true" ]; then
    validate_positive_int "$PF_N" "PF_N"
fi

read -r -a ENSEMBLE_SIZES <<< "${ENSEMBLE_SIZES_RAW//,/ }"
if [ "${#ENSEMBLE_SIZES[@]}" -eq 0 ]; then
    echo "Error: ENSEMBLE_SIZES is empty." >&2
    exit 1
fi
for n in "${ENSEMBLE_SIZES[@]}"; do
    validate_positive_int "$n" "ENSEMBLE_SIZES entry"
done

METHODS_TOKENS="${METHODS_RAW//,/ }"
METHODS_TOKENS="${METHODS_TOKENS//+/ }"
read -r -a METHOD_LIST <<< "$METHODS_TOKENS"
if [ "${#METHOD_LIST[@]}" -eq 0 ]; then
    echo "Error: METHODS is empty." >&2
    exit 1
fi

echo "Date: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Host: $(hostname)"
echo "Python: $PYTHON_BIN"
echo "DATASET: $DATASET"
echo "SIGMA_Y: $SIGMA_Y"
echo "OBS_FN: $OBS_FN"
echo "METHODS: ${METHOD_LIST[*]}"
echo "SEED: $SEED"
echo "ENSEMBLE_SIZES: ${ENSEMBLE_SIZES[*]}"
echo "DEVICE: $DEVICE"
echo "NUM_LOADER_WORKERS: $NUM_LOADER_WORKERS"
echo "ADAPTIVE_SIGMA_Y: $ADAPTIVE_SIGMA_Y"
echo "USE_LOCALIZATION: $USE_LOCALIZATION"
echo "NORMAL_OUTPUT: $NORMAL_OUTPUT"
echo "SAVE_TEST_FIGURES: $SAVE_TEST_FIGURES"
echo "PF_VERIFICATION: $PF_VERIFICATION"
echo "PF_N: $PF_N"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-unset}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "----------------------------------------------------"

for N in "${ENSEMBLE_SIZES[@]}"; do
    for method in "${METHOD_LIST[@]}"; do
        echo "============================================================"
        echo "Running benchmark test: dataset=$DATASET obs_fn=$OBS_FN N=$N method=$method"
        echo "============================================================"

        cmd=(
            "$PYTHON_BIN" evaluate_benchmark.py
            --device "$DEVICE"
            --dataset "$DATASET"
            --N "$N"
            --sigma_y "$SIGMA_Y"
            --seed "$SEED"
            --v "$method"
            --obs_fn "$OBS_FN"
            --num_loader_workers "$NUM_LOADER_WORKERS"
        )

        if [ "$USE_LOCALIZATION" != "true" ]; then
            cmd+=(--no_localization)
        fi
        if [ "$NORMAL_OUTPUT" = "true" ]; then
            cmd+=(--normal_output)
        fi
        if [ "$SAVE_TEST_FIGURES" = "true" ]; then
            cmd+=(--save_test_figures)
        fi
        if [ "$ADAPTIVE_SIGMA_Y" = "true" ]; then
            cmd+=(--adaptive_sigma_y)
        fi
        if [ "$PF_VERIFICATION" = "true" ]; then
            cmd+=(--pf_verification --pf_N "$PF_N")
        fi

        "${cmd[@]}"

        echo "Finished N=$N method=$method."
        echo ""
    done
done

echo "All benchmark tests finished."
echo "Job completed on $(date)"
