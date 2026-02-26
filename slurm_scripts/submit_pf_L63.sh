#!/bin/bash

# Description: Submit active PF-L63 experiments from scripts/run_pf_results.sh.
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
# DRAW_FIGURE OBS_FN ADAPTIVE_SIGMA_Y PF_N
# PF_N accepts a positive integer or "auto".
EXPERIMENTS=(
    "true square true 10000"
    "true arctan true 10000"
    "true default true 10000"
    "true square true 1000000"
    "true arctan true 1000000"
    "true default true 1000000"
    "false square true auto"
    "false arctan true auto"
    "false default true auto"
)

validate_pf_n() {
    local v="$1"
    if [ "$v" = "auto" ]; then
        return 0
    fi

    if [[ "$v" =~ ^[1-9][0-9]*$ ]]; then
        return 0
    fi

    echo "Error: PF_N must be a positive integer or 'auto', got '$v'." >&2
    return 1
}

for exp in "${EXPERIMENTS[@]}"; do
    read -r draw_figure obs_fn adaptive_sigma_y pf_n <<< "$exp"

    validate_bool "$draw_figure"
    validate_bool "$adaptive_sigma_y"
    validate_pf_n "$pf_n"

    if [ "$draw_figure" = "true" ]; then
        mode="vis"
    else
        mode="eval"
    fi

    job_name="pf-l63-${obs_fn}-${mode}"
    if [ "$pf_n" != "auto" ]; then
        job_name="${job_name}-n${pf_n}"
    fi

    echo "Submitting: $job_name (DRAW_FIGURE=$draw_figure, OBS_FN=$obs_fn, ADAPTIVE_SIGMA_Y=$adaptive_sigma_y, PF_N=$pf_n)"

    sbatch -J "$job_name" \
        --export=ALL,DRAW_FIGURE="$draw_figure",OBS_FN="$obs_fn",ADAPTIVE_SIGMA_Y="$adaptive_sigma_y",PF_N="$pf_n" \
        "$SLURM_SCRIPT"
done
