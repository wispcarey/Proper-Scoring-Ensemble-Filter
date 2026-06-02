#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-lorenz63}"
MODEL="${MODEL:-EtE-LRes}"
LOSS="${LOSS:-es}"
ES_P="${ES_P:-1}"
N="${N:-10}"
EPOCHS="${EPOCHS:-1}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
SIGMA_Y="${SIGMA_Y:-1.0}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SAVE_EPOCH="${SAVE_EPOCH:-25}"
LOCALIZATION="${LOCALIZATION:-auto}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
DRY_RUN="${DRY_RUN:-false}"

cmd=(
  "$PYTHON_BIN" train.py
  --dataset "$DATASET"
  --v "$MODEL"
  --loss_type "$LOSS"
  --N "$N"
  --epochs "$EPOCHS"
  --device "$DEVICE"
  --seed "$SEED"
  --sigma_y "$SIGMA_Y"
  --num_loader_workers "$NUM_WORKERS"
  --save_epoch "$SAVE_EPOCH"
  --normal_output
  --no_running_loss
)

if [[ ",$LOSS," == *",es,"* ]]; then
  cmd+=(--es_p "$ES_P")
fi

if [[ "$LOCALIZATION" == "false" || "$DATASET" == "lorenz63" || "$DATASET" == "doubling1d" ]]; then
  cmd+=(--no_localization)
fi

if [[ -n "$EXTRA_ARGS" ]]; then
  # Optional user-supplied extra flags, e.g. EXTRA_ARGS="--obs_fn square".
  cmd+=($EXTRA_ARGS)
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "$DRY_RUN" == "true" ]]; then
  exit 0
fi

"${cmd[@]}"
