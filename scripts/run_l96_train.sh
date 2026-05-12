#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT" || exit 1

PYTHON_BIN="/home/bhchen/miniconda3/bin/python"

# Local Lorenz96 training runner mirroring the active high-dimensional
# experiment list, without cluster-only settings.
DATASET="lorenz96"
SEED="${SEED:-42}"
NUM_LOADER_WORKERS="${NUM_LOADER_WORKERS:-1}"
SAVE_EPOCH="${SAVE_EPOCH:-10}"
ES_P="${ES_P:-1}"

# Columns:
#   epochs N sigma_y version loss_type loss_weights learning_rate obs_fn weight_decay adaptive_sigma_y suffix
EXPERIMENTS=(
    # "1000 10 1.0 EtE-LRes es None default default 0 true None"
    # "1000 10 1.0 EtE-LRes nl2 None default default 0 true None"
    # "1000 10 1.0 CorrTerms es None default default 0.01 true None"
    "100 10 1.0 CorrTerms nl2 None default default 0.01 true None"
)

validate_adaptive_sigma_y() {
    local value="$1"
    case "$value" in
        true|false) return 0 ;;
        *)
            echo "Error: adaptive_sigma_y must be 'true' or 'false'. Got '$value'" >&2
            return 1
            ;;
    esac
}

run_experiment() {
    local epochs="$1"
    local n="$2"
    local sigma_y="$3"
    local version="$4"
    local loss_type="$5"
    local loss_weights="$6"
    local learning_rate="$7"
    local obs_fn="$8"
    local weight_decay="$9"
    local adaptive_sigma_y="${10}"
    local suffix="${11}"

    validate_adaptive_sigma_y "$adaptive_sigma_y"

    local adaptive_sigma_y_flag=()
    if [ "$adaptive_sigma_y" = "true" ]; then
        adaptive_sigma_y_flag=(--adaptive_sigma_y)
    fi

    local suffix_flag=()
    if [ "$suffix" != "None" ] && [ -n "$suffix" ]; then
        suffix_flag=(--suffix "$suffix")
    fi

    echo "=================================================="
    echo "Training L96: v=$version loss=$loss_type N=$n obs_fn=$obs_fn lr=$learning_rate wd=$weight_decay adaptive_sigma_y=$adaptive_sigma_y"
    echo "=================================================="

    "$PYTHON_BIN" train.py \
        --dataset "$DATASET" \
        --num_loader_workers "$NUM_LOADER_WORKERS" \
        --epochs "$epochs" \
        --N "$n" \
        --sigma_y "$sigma_y" \
        "${adaptive_sigma_y_flag[@]}" \
        --seed "$SEED" \
        --v "$version" \
        --learning_rate "$learning_rate" \
        --obs_fn "$obs_fn" \
        --weight_decay "$weight_decay" \
        "${suffix_flag[@]}" \
        --loss_type "$loss_type" \
        --loss_weights "$loss_weights" \
        --es_p "$ES_P" \
        --save_epoch "$SAVE_EPOCH" \
        --normal_output
}

for exp in "${EXPERIMENTS[@]}"; do
    read -r epochs n sigma_y version loss_type loss_weights learning_rate obs_fn weight_decay adaptive_sigma_y suffix <<< "$exp"
    run_experiment \
        "$epochs" \
        "$n" \
        "$sigma_y" \
        "$version" \
        "$loss_type" \
        "$loss_weights" \
        "$learning_rate" \
        "$obs_fn" \
        "$weight_decay" \
        "$adaptive_sigma_y" \
        "$suffix"
done
