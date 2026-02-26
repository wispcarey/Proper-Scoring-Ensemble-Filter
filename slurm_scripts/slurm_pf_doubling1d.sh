#!/bin/bash

# Submit this script with: sbatch <this-filename>

#SBATCH --time=1-00:00:00     # walltime (1 days)
#SBATCH --nodes=1             # number of nodes (1 node)
#SBATCH --gres=gpu:1          # 1 GPU
#SBATCH --partition=gpu       # use GPU partition
#SBATCH --ntasks=1            # 1 task
#SBATCH -J "bpf-doubling1d"   # job name
#SBATCH --mail-user=bhchen@caltech.edu # email address
#SBATCH --mail-type=BEGIN     # email notification at start
#SBATCH --mail-type=END       # email notification at end
#SBATCH --mail-type=FAIL      # email notification on failure

# Optional: specify output and error files
#SBATCH -o slurm.%N.%j.out    # STDOUT
#SBATCH -e slurm.%N.%j.err    # STDERR

module load cuda/12.2

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

parse_pf_n() {
    local raw="${1:-auto}"
    local v
    v="$(echo "$raw" | tr '[:upper:]' '[:lower:]')"
    case "$v" in
        auto|"") echo "auto" ;;
        *[!0-9]*)
            echo "Error: invalid PF_N '$raw' (must be positive integer or 'auto')." >&2
            exit 1
            ;;
        0)
            echo "Error: PF_N must be > 0, got '$raw'." >&2
            exit 1
            ;;
        *)
            echo "$v"
            ;;
    esac
}

# Input parameters (can be overridden by sbatch --export)
DRAW_FIGURE="$(parse_bool "${DRAW_FIGURE:-false}")"
PF_N="$(parse_pf_n "${PF_N:-auto}")"
PF_SAVE_FIGURE="$DRAW_FIGURE"

echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Python: $(command -v "$PYTHON_BIN")"
echo "DRAW_FIGURE: $DRAW_FIGURE"
echo "PF_SAVE_FIGURE: $PF_SAVE_FIGURE"
echo "PF_N: $PF_N"
echo "----------------------------------------------------"

if [ "$DRAW_FIGURE" = "true" ]; then
    # Visualization experiment in scripts/run_pf_results.sh
    SEEDS=(42)
    if [ "$PF_N" = "auto" ]; then
        PARTICLE_NUMBERS=(1000000)
    else
        PARTICLE_NUMBERS=("$PF_N")
    fi
else
    # Evaluation sweep (optional)
    SEEDS=(0 1 2 3 4 5 6 7 8 9 10 42)
    if [ "$PF_N" = "auto" ]; then
        PARTICLE_NUMBERS=(500 1000 2000 5000 10000 20000 50000 100000 200000 500000 1000000)
    else
        PARTICLE_NUMBERS=("$PF_N")
    fi
fi

for seed_val in "${SEEDS[@]}"; do
    for pf_n_val in "${PARTICLE_NUMBERS[@]}"; do
        echo "============================================================"
        echo "Running with Seed: $seed_val and Particle Count (pf_N): $pf_n_val"
        echo "============================================================"

        cmd=(
            "$PYTHON_BIN" gen_pf_results.py
            --dataset doubling1d
            --seed "$seed_val"
            --normal_output
            --test_steps 200
            --pf_verification
            --pf_N "$pf_n_val"
            --sigma_reg None
        )

        if [ "$PF_SAVE_FIGURE" = "true" ]; then
            cmd+=(--pf_save_figure)
        fi

        "${cmd[@]}"

        echo "Done."
        echo ""
    done
done

echo "All experiments finished."
echo "Job completed on $(date)"
