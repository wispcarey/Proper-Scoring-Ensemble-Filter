#!/bin/bash

# Submit this script with:
# sbatch --export=ALL,DATASET=lorenz63,METHOD=EnKF,OBS_FN=square slurm_grid_search_benchmark.sh

#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH -J "grid-bench"
#SBATCH --mail-user=bhchen@caltech.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    else
        echo "Error: neither 'python' nor 'python3' is available in PATH." >&2
        exit 127
    fi
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
METHOD="${METHOD:-EnKF}"
OBS_FN="${OBS_FN:-square}"
SEED="${SEED:-42}"
ENSEMBLE_SIZES_RAW="${ENSEMBLE_SIZES:-5 10 15 20 40 60 100}"
GRID_SEARCH_NUM_SEEDS="${GRID_SEARCH_NUM_SEEDS:-4}"
CPU_WORKERS="${CPU_WORKERS:-64}"
ADAPTIVE_SIGMA_Y="$(parse_bool "${ADAPTIVE_SIGMA_Y:-true}")"

case "$DATASET" in
    lorenz63|doubling1d)
        DEFAULT_DISABLE_LOCALIZATION="true"
        DEFAULT_USE_PF_VERIFICATION="true"
        DEFAULT_PF_N="1000000"
        ;;
    *)
        DEFAULT_DISABLE_LOCALIZATION="false"
        DEFAULT_USE_PF_VERIFICATION="false"
        DEFAULT_PF_N="1000000"
        ;;
esac

DISABLE_LOCALIZATION="$(parse_bool "${DISABLE_LOCALIZATION:-$DEFAULT_DISABLE_LOCALIZATION}")"
USE_PF_VERIFICATION="$(parse_bool "${USE_PF_VERIFICATION:-$DEFAULT_USE_PF_VERIFICATION}")"
PF_N="${PF_N:-$DEFAULT_PF_N}"

validate_positive_int "$SEED" "SEED"
validate_positive_int "$GRID_SEARCH_NUM_SEEDS" "GRID_SEARCH_NUM_SEEDS"
validate_positive_int "$CPU_WORKERS" "CPU_WORKERS"
if [ "$USE_PF_VERIFICATION" = "true" ]; then
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

REQUESTED_CPUS="${SLURM_CPUS_PER_TASK:-64}"
if [ "$CPU_WORKERS" -gt "$REQUESTED_CPUS" ]; then
    echo "Error: CPU_WORKERS=$CPU_WORKERS exceeds SLURM_CPUS_PER_TASK=$REQUESTED_CPUS." >&2
    exit 1
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

echo "Date: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Host: $(hostname)"
echo "Python: $(command -v "$PYTHON_BIN")"
echo "DATASET: $DATASET"
echo "METHOD: $METHOD"
echo "OBS_FN: $OBS_FN"
echo "SEED: $SEED"
echo "ENSEMBLE_SIZES: ${ENSEMBLE_SIZES[*]}"
echo "ADAPTIVE_SIGMA_Y: $ADAPTIVE_SIGMA_Y"
echo "DISABLE_LOCALIZATION: $DISABLE_LOCALIZATION"
echo "USE_PF_VERIFICATION: $USE_PF_VERIFICATION"
echo "PF_N: $PF_N"
echo "CPU_WORKERS: $CPU_WORKERS"
echo "GRID_SEARCH_NUM_SEEDS: $GRID_SEARCH_NUM_SEEDS"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-unset}"
echo "----------------------------------------------------"

for N in "${ENSEMBLE_SIZES[@]}"; do
    echo "============================================================"
    echo "Running grid search: dataset=$DATASET obs_fn=$OBS_FN N=$N method=$METHOD"
    echo "============================================================"

    cmd=(
        "$PYTHON_BIN" grid_search_benchmark.py
        --device cpu
        --dataset "$DATASET"
        --N "$N"
        --seed "$SEED"
        --v "$METHOD"
        --obs_fn "$OBS_FN"
        --grid_search_cpu_workers "$CPU_WORKERS"
        --grid_search_num_seeds "$GRID_SEARCH_NUM_SEEDS"
        --disable_tqdm
        --normal_output
    )

    if [ "$ADAPTIVE_SIGMA_Y" = "true" ]; then
        cmd+=(--adaptive_sigma_y)
    fi
    if [ "$DISABLE_LOCALIZATION" = "true" ]; then
        cmd+=(--no_localization)
    fi
    if [ "$USE_PF_VERIFICATION" = "true" ]; then
        cmd+=(--pf_verification --pf_N "$PF_N")
    fi

    "${cmd[@]}"

    echo "Finished N=$N."
    echo ""
done

echo "All N runs finished."
echo "Job completed on $(date)"
