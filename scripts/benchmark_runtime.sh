#!/usr/bin/env bash
# set -x  # Uncomment for debugging
set -euo pipefail

# ---------------------------------------------
# Usage:
#   ./run_eval.sh --device cuda            # default
#   ./run_eval.sh --device cpu
#   ./run_eval.sh --seed 42                # optional
#   ./run_eval.sh --dry-run                # only print commands
#   ./run_eval.sh --dataset lorenz96       # limit to one dataset (lorenz96|ks)
#   ./run_eval.sh --methods EnKF,ESRF      # limit methods (comma-separated)
# ---------------------------------------------

# ---------- Defaults ----------
DEVICE="cuda"
SEED=42
DRY_RUN=0
ONLY_DATASET=""     # empty = run both lorenz96 and ks
ONLY_METHODS=""     # empty = use the predefined groups below

# ---------- Parse args ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)
      DEVICE="${2:-}"; shift 2;;
    --seed)
      SEED="${2:-}"; shift 2;;
    --dry-run)
      DRY_RUN=1; shift;;
    --dataset)
      ONLY_DATASET="${2:-}"; shift 2;;
    --methods)
      ONLY_METHODS="${2:-}"; shift 2;;
    -h|--help)
      sed -n '1,80p' "$0"; exit 0;;
    *)
      echo "Unknown argument: $1" >&2; exit 1;;
  esac
done

# ---------- Resolve project root and cd there ----------
# Moves to repo root (one level above this script), matching your original `cd ..`
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

# ---------- Experiment grids ----------
Ns=(5 10 15 20 40 60 100)
SIGMAS=(1 0.7)

# Methods that REQUIRE localization (DO NOT pass --no_localization)
METHODS_LOC_DEFAULT=( "EnKF" "LETKF" )
# Methods that REQUIRE NO localization (MUST pass --no_localization)
METHODS_NOLOC_DEFAULT=( "ESRF" "iEnKS-PertObs" "iEnKS-Sqrt" "iEnKS-Order1" )

# If user specified --methods, override both groups and infer loc flag per method later
IFS=',' read -r -a ONLY_METHODS_ARR <<< "${ONLY_METHODS:-}"

# ---------- Helper: check if a method is in a list ----------
in_array() {
  local needle="$1"; shift
  local e
  for e in "$@"; do
    [[ "$e" == "$needle" ]] && return 0
  done
  return 1
}

# ---------- Helper: build and run a single command ----------
run_cmd() {
  # $1 dataset, $2 N, $3 sigma_y, $4 method, $5 device, $6 no_loc_flag (0/1), $7 seed
  local dataset="$1" N="$2" sigma="$3" method="$4" device="$5" no_loc="$6" seed="$7"

  # Build base command
  cmd=( python evaluate_benchmark.py
    --device "$device"
    --dataset "$dataset"
    --N "$N"
    --sigma_y "$sigma"
    --seed "$seed"
    --v "$method"
    --normal_output
  )

  # Add --no_localization if needed
  if [[ "$no_loc" -eq 1 ]]; then
    cmd+=( --no_localization )
  fi

  echo "Running: dataset=${dataset} method=${method} N=${N} sigma_y=${sigma} device=${device} no_loc=${no_loc}"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf 'CMD: %q ' "${cmd[@]}"; echo
  else
    "${cmd[@]}"
  fi
}

# ---------- Helper: run a suite for a dataset ----------
run_dataset_suite() {
  # $1 dataset name
  local dataset="$1"

  # Decide method lists
  local methods_loc=()
  local methods_noloc=()

  if [[ -n "$ONLY_METHODS" ]]; then
    # Split and route methods according to known localization rule:
    # - EnKF, LETKF => need localization (no --no_localization)
    # - ESRF, any iEnKS-* => need --no_localization
    local m
    for m in "${ONLY_METHODS_ARR[@]}"; do
      if [[ "$m" == "EnKF" || "$m" == "LETKF" ]]; then
        methods_loc+=( "$m" )
      else
        # ESRF or iEnKS-* fall here
        methods_noloc+=( "$m" )
      fi
    done
  else
    # Defaults: you can edit these based on what you want to run by default per dataset
    # Keep both groups present to mirror original intent
    methods_loc=( "${METHODS_LOC_DEFAULT[@]}" )
    methods_noloc=( "${METHODS_NOLOC_DEFAULT[@]}" )
  fi

  # Iterate grids
  local sigma N method
  for sigma in "${SIGMAS[@]}"; do
    for N in "${Ns[@]}"; do
      # with localization
      for method in "${methods_loc[@]}"; do
        run_cmd "$dataset" "$N" "$sigma" "$method" "$DEVICE" 0 "$SEED"
      done
      # no localization
      for method in "${methods_noloc[@]}"; do
        run_cmd "$dataset" "$N" "$sigma" "$method" "$DEVICE" 1 "$SEED"
      done
    done
  done
}

# ---------- Entry ----------
# If user constrained dataset, run only that; otherwise run both to match the original script
case "${ONLY_DATASET:-}" in
  "")
    run_dataset_suite "lorenz96"
    run_dataset_suite "ks"
    ;;
  "lorenz96"|"ks")
    run_dataset_suite "$ONLY_DATASET"
    ;;
  *)
    echo "Unknown dataset: $ONLY_DATASET (expected: lorenz96|ks)" >&2; exit 1;;
esac
