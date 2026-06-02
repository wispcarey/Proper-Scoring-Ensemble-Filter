#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-lorenz63}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
SIGMA_Y="${SIGMA_Y:-1.0}"
OBS_FN="${OBS_FN:-default}"
PF_N="${PF_N:-1000}"
PF_SAVE_FIGURE="${PF_SAVE_FIGURE:-false}"
LOCALIZATION="${LOCALIZATION:-auto}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
DRY_RUN="${DRY_RUN:-false}"

cmd=(
  "$PYTHON_BIN" gen_pf_results.py
  --dataset "$DATASET"
  --device "$DEVICE"
  --seed "$SEED"
  --sigma_y "$SIGMA_Y"
  --obs_fn "$OBS_FN"
  --pf_N "$PF_N"
  --normal_output
)

if [[ "$PF_SAVE_FIGURE" == "true" ]]; then
  cmd+=(--pf_save_figure)
fi

if [[ "$LOCALIZATION" == "false" || "$DATASET" == "lorenz63" || "$DATASET" == "doubling1d" ]]; then
  cmd+=(--no_localization)
fi

if [[ -n "$EXTRA_ARGS" ]]; then
  cmd+=($EXTRA_ARGS)
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "$DRY_RUN" == "true" ]]; then
  exit 0
fi

"${cmd[@]}"
