#!/bin/bash
# Run each setting twice: once on CPU and once on CUDA (GPU).
# Assumes evaluate.py supports --device cpu|cuda.

set -e

cd ..

run_eval () {
  local dataset="$1"
  local sigma_y="$2"
  local seed="$3"

  # You can adjust N_LIST below if needed
  local N_LIST=("5" "10" "15" "20" "40" "60" "100")

  for device in cpu cuda; do
    echo "=== Running dataset=${dataset}, sigma_y=${sigma_y}, seed=${seed}, device=${device} ==="
    for N in "${N_LIST[@]}"; do
      echo "---- N=${N} ----"
      python evaluate.py \
        --device "${device}" \
        --dataset "${dataset}" \
        --N "${N}" \
        --sigma_y "${sigma_y}" \
        --seed "${seed}" \
        --normal_output \
        --v CorrTerms
    done
  done
}

# lorenz96 block
# dataset="lorenz96"
# sigma_y=1
# seed=42
# run_eval "${dataset}" "${sigma_y}" "${seed}"

# ks block
dataset="ks"
sigma_y=1
seed=42
run_eval "${dataset}" "${sigma_y}" "${seed}"
