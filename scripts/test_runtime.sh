#!/usr/bin/env bash
set -euo pipefail

# Runtime-only test suite for Lorenz-96 with the dataset-default observation
# function. The raw evaluator outputs keep their existing save paths; this script
# copies each run's records/log into an isolated archive directory afterward.

PYTHON_BIN="/home/bhchen/miniconda3/bin/python"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATASET="lorenz96"
OBS_FN="default"
SIGMA_Y="1.0"
SEED="42"
N_LIST=(5 10 15 20 40 60 100)
ML_METHODS=("CorrTerms" "EtE-LRes")
ML_DEVICES=("cpu" "cuda")
CLASSIC_METHODS=("EnKF" "LETKF" "iEnKS-PertObs")
CLASSIC_DEVICE="cpu"

REQUIRES_CUDA=0
for device in "${ML_DEVICES[@]}"; do
  if [[ "${device}" == "cuda" ]]; then
    REQUIRES_CUDA=1
  fi
done
if [[ "${REQUIRES_CUDA}" -eq 1 ]]; then
  if ! "${PYTHON_BIN}" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
    printf "Error: ML_DEVICES includes cuda, but PyTorch does not see an available CUDA device.\n" >&2
    exit 1
  fi
fi

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUNTIME_ROOT="${RUNTIME_ROOT:-save/runtime_test/${DATASET}_${RUN_TAG}}"
MANIFEST="${RUNTIME_ROOT}/manifest.tsv"

mkdir -p "${RUNTIME_ROOT}/ml" "${RUNTIME_ROOT}/classic"
printf "kind\tmethod\tdevice\tN\tsigma_y\tobs_fn\tstatus\toutput_dir\tarchive_dir\n" > "${MANIFEST}"

current_log_lines() {
  local output_dir="$1"
  local log_file="${output_dir}/test_output.txt"

  if [[ -f "${log_file}" ]]; then
    wc -l < "${log_file}"
  else
    printf "0\n"
  fi
}

ml_output_device_label() {
  local requested_device="$1"

  if [[ "${requested_device}" == "cuda" ]]; then
    printf "cuda:0\n"
  else
    printf "%s\n" "${requested_device}"
  fi
}

copy_run_artifacts() {
  local output_dir="$1"
  local archive_dir="$2"
  local output_record="$3"
  local log_start_line="$4"
  local log_file="${output_dir}/test_output.txt"

  mkdir -p "${archive_dir}"
  if [[ -f "${output_record}" ]]; then
    cp "${output_record}" "${archive_dir}/"
  else
    printf "Warning: expected output record not found: %s\n" "${output_record}" >&2
  fi

  if [[ -f "${log_file}" ]]; then
    tail -n +"$((log_start_line + 1))" "${log_file}" > "${archive_dir}/test_output.txt"
  fi
}

run_ml_eval() {
  local method="$1"
  local device="$2"
  local N="$3"
  local output_device
  local suffix="_runtime_${RUN_TAG}_${method}_${device}"
  output_device="$(ml_output_device_label "${device}")"
  local output_dir="save/benchmark_${DATASET}_${SIGMA_Y}_${method}_${output_device}"
  local archive_dir="${RUNTIME_ROOT}/ml/${method}/${device}/N${N}"
  local output_record="${output_dir}/output_records_${N}${suffix}.pt"
  local log_start_line

  printf "\n=== ML runtime: method=%s device=%s N=%s ===\n" "${method}" "${device}" "${N}"
  log_start_line="$(current_log_lines "${output_dir}")"
  "${PYTHON_BIN}" evaluate.py \
    --device "${device}" \
    --dataset "${DATASET}" \
    --N "${N}" \
    --sigma_y "${SIGMA_Y}" \
    --seed "${SEED}" \
    --obs_fn "${OBS_FN}" \
    --v "${method}" \
    --cp_load_path no \
    --suffix "${suffix}"

  copy_run_artifacts "${output_dir}" "${archive_dir}" "${output_record}" "${log_start_line}"
  printf "ml\t%s\t%s\t%s\t%s\t%s\tok\t%s\t%s\n" \
    "${method}" "${device}" "${N}" "${SIGMA_Y}" "${OBS_FN}" "${output_dir}" "${archive_dir}" >> "${MANIFEST}"
}

classic_needs_no_localization() {
  local method="$1"
  [[ "${method}" == "iEnKS-PertObs" ]]
}

run_classic_eval() {
  local method="$1"
  local N="$2"
  local suffix="_runtime_${RUN_TAG}_${method}_${CLASSIC_DEVICE}"
  local output_dir="save/benchmark_${DATASET}_${SIGMA_Y}_${method}"
  local archive_dir="${RUNTIME_ROOT}/classic/${method}/${CLASSIC_DEVICE}/N${N}"
  local output_record="${output_dir}/output_records_${N}${suffix}.pt"
  local log_start_line
  local cmd=(
    "${PYTHON_BIN}" evaluate_benchmark.py
    --device "${CLASSIC_DEVICE}"
    --dataset "${DATASET}"
    --N "${N}"
    --sigma_y "${SIGMA_Y}"
    --seed "${SEED}"
    --obs_fn "${OBS_FN}"
    --v "${method}"
    --suffix "${suffix}"
  )

  if classic_needs_no_localization "${method}"; then
    cmd+=(--no_localization)
  fi

  printf "\n=== Classic runtime: method=%s device=%s N=%s ===\n" "${method}" "${CLASSIC_DEVICE}" "${N}"
  log_start_line="$(current_log_lines "${output_dir}")"
  "${cmd[@]}"

  copy_run_artifacts "${output_dir}" "${archive_dir}" "${output_record}" "${log_start_line}"
  printf "classic\t%s\t%s\t%s\t%s\t%s\tok\t%s\t%s\n" \
    "${method}" "${CLASSIC_DEVICE}" "${N}" "${SIGMA_Y}" "${OBS_FN}" "${output_dir}" "${archive_dir}" >> "${MANIFEST}"
}

for N in "${N_LIST[@]}"; do
  for method in "${ML_METHODS[@]}"; do
    for device in "${ML_DEVICES[@]}"; do
      run_ml_eval "${method}" "${device}" "${N}"
    done
  done

  for method in "${CLASSIC_METHODS[@]}"; do
    run_classic_eval "${method}" "${N}"
  done
done

printf "\nRuntime archive saved to: %s\n" "${RUNTIME_ROOT}"
printf "Manifest: %s\n" "${MANIFEST}"
