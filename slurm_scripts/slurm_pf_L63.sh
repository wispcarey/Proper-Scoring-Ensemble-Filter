#!/bin/bash

# Submit this script with: sbatch <this-filename>

#SBATCH --time=1-00:00:00     # walltime (1 days)
#SBATCH --nodes=1           # number of nodes (1 node)
#SBATCH --gres=gpu:1        # 4 GPUs of any type
#SBATCH --partition=gpu     # use GPU partition
#SBATCH --ntasks=1          # 1 task
#SBATCH -J "bpf-l63"   # job name
#SBATCH --mail-user=bhchen@caltech.edu # email address
#SBATCH --mail-type=BEGIN   # email notification at start
#SBATCH --mail-type=END     # email notification at end
#SBATCH --mail-type=FAIL    # email notification on failure

# Optional: specify output and error files
#SBATCH -o slurm.%N.%j.out  # STDOUT
#SBATCH -e slurm.%N.%j.err  # STDERR

# Load modules if necessary (e.g., CUDA or other dependencies)
module load cuda/12.2  # Adjusted to CUDA version 12.2

# Change to repo root
cd ..

set -e

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

# Input parameters (can be overridden by sbatch --export)
DRAW_FIGURE="$(parse_bool "${DRAW_FIGURE:-false}")"
PF_SAVE_FIGURE="$(parse_bool "${PF_SAVE_FIGURE:-false}")"
OBS_FN="${OBS_FN:-square}"
ADAPTIVE_SIGMA_Y="$(parse_bool "${ADAPTIVE_SIGMA_Y:-true}")"

echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Python: $(command -v "$PYTHON_BIN")"
echo "DRAW_FIGURE: $DRAW_FIGURE"
echo "PF_SAVE_FIGURE: $PF_SAVE_FIGURE"
echo "OBS_FN: $OBS_FN"
echo "ADAPTIVE_SIGMA_Y: $ADAPTIVE_SIGMA_Y"
echo "----------------------------------------------------"

if [ "$DRAW_FIGURE" = "true" ]; then
    # Visualization experiments in scripts/run_pf_results.sh
    SEEDS=(42)
    PARTICLE_NUMBERS=(1000000)
else
    # Non-visualization experiments in scripts/run_pf_results.sh
    SEEDS=(0 1 2 3 4 5 6 7 8 9 10 42)
    PARTICLE_NUMBERS=(500 1000 2000 5000 10000 20000 50000 100000 200000 500000 1000000)
fi

for seed_val in "${SEEDS[@]}"; do
    for pf_n_val in "${PARTICLE_NUMBERS[@]}"; do
        
        echo "============================================================"
        echo "Running with Seed: $seed_val and Particle Count (pf_N): $pf_n_val"
        echo "============================================================"
        
        cmd=(
            "$PYTHON_BIN" gen_pf_results.py
            --dataset lorenz63
            --seed "$seed_val"
            --normal_output
            --test_steps 500
            --pf_verification
            --pf_N "$pf_n_val"
            --sigma_reg None
            --obs_fn "$OBS_FN"
        )

        if [ "$PF_SAVE_FIGURE" = "true" ]; then
            cmd+=(--pf_save_figure)
        fi
        if [ "$ADAPTIVE_SIGMA_Y" = "true" ]; then
            cmd+=(--adaptive_sigma_y)
        fi

        "${cmd[@]}"
        
        echo "Done."
        echo ""
    done
done

echo "All experiments finished."
echo "Job completed on $(date)"
