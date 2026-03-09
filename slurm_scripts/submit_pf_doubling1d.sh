#!/bin/bash

# Description: Submit PF-doubling1d experiments from scripts/run_pf_results.sh.
# Usage: bash submit_pf_doubling1d.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="$SCRIPT_DIR/slurm_pf_doubling1d.sh"
GPU_TYPE="${1:-p100}"

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

# Format per entry:
# DRAW_FIGURE PF_N
# PF_N accepts a positive integer or "auto".
EXPERIMENTS=(
    "true 1000000"
    "true 10000"
    "false auto"
)

for exp in "${EXPERIMENTS[@]}"; do
    read -r draw_figure pf_n <<< "$exp"

    validate_bool "$draw_figure"
    validate_pf_n "$pf_n"

    if [ "$draw_figure" = "true" ]; then
        mode="vis"
    else
        mode="eval"
    fi

    job_name="pf-doubling1d-${mode}"
    if [ "$pf_n" != "auto" ]; then
        job_name="${job_name}-n${pf_n}"
    fi

    echo "Submitting: $job_name (DRAW_FIGURE=$draw_figure, PF_N=$pf_n)"

    sbatch -J "$job_name" \
        --gres="gpu:${GPU_TYPE}:1" \
        --export=ALL,DRAW_FIGURE="$draw_figure",PF_N="$pf_n" \
        "$SLURM_SCRIPT"
done
