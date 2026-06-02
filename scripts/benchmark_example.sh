#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-lorenz63}"
METHOD="${METHOD:-ESRF}"
N="${N:-10}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
SIGMA_Y="${SIGMA_Y:-1.0}"
OBS_FN="${OBS_FN:-default}"
LOCALIZATION="${LOCALIZATION:-auto}"
PF_VERIFICATION="${PF_VERIFICATION:-false}"
PF_VERIFICATION_SEED="${PF_VERIFICATION_SEED:-42}"
PF_N="${PF_N:-1000}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
DRY_RUN="${DRY_RUN:-false}"

cmd=(
  "$PYTHON_BIN" evaluate_benchmark.py
  --dataset "$DATASET"
  --v "$METHOD"
  --N "$N"
  --device "$DEVICE"
  --seed "$SEED"
  --sigma_y "$SIGMA_Y"
  --obs_fn "$OBS_FN"
  --normal_output
)

if [[ "$PF_VERIFICATION" == "true" ]]; then
  cmd+=(--pf_verification --pf_verification_seed "$PF_VERIFICATION_SEED" --pf_N "$PF_N")
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
