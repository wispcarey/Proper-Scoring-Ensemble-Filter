#!/bin/bash

# Description: Submit the 4 active PF-L63 experiments from scripts/run_pf_results.sh.
# Usage: bash submit_pf_L63.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="$SCRIPT_DIR/slurm_pf_L63.sh"

validate_bool() {
    local v="$1"
    case "$v" in
        true|false) return 0 ;;
        *)
            echo "Error: boolean value must be 'true' or 'false', got '$v'." >&2
            return 1
            ;;
    esac
}

# Format per entry:
# DRAW_FIGURE PF_SAVE_FIGURE OBS_FN ADAPTIVE_SIGMA_Y
EXPERIMENTS=(
    "true true square true"
    "true true arctan true"
    "false false square true"
    "false false arctan true"
)

for exp in "${EXPERIMENTS[@]}"; do
    read -r draw_figure pf_save_figure obs_fn adaptive_sigma_y <<< "$exp"

    validate_bool "$draw_figure"
    validate_bool "$pf_save_figure"
    validate_bool "$adaptive_sigma_y"

    if [ "$draw_figure" = "true" ]; then
        mode="vis"
    else
        mode="eval"
    fi

    job_name="pf-l63-${obs_fn}-${mode}"

    echo "Submitting: $job_name (DRAW_FIGURE=$draw_figure, PF_SAVE_FIGURE=$pf_save_figure, OBS_FN=$obs_fn, ADAPTIVE_SIGMA_Y=$adaptive_sigma_y)"

    sbatch -J "$job_name" \
        --export=ALL,DRAW_FIGURE="$draw_figure",PF_SAVE_FIGURE="$pf_save_figure",OBS_FN="$obs_fn",ADAPTIVE_SIGMA_Y="$adaptive_sigma_y" \
        "$SLURM_SCRIPT"
done
