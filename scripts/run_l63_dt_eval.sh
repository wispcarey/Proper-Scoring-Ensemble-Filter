#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT" || exit 1

PYTHON_BIN="${PYTHON_BIN:-/home/bhchen/miniconda3/bin/python}"
dataset="lorenz63"
seed="${SEED:-42}"
results_root="${RESULTS_ROOT:-save/lorenz63_results}"
pf_N="${PF_N:-1000000}"
save_test_figures="${SAVE_TEST_FIGURES:-0}"

plot_start_step="${PLOT_START_STEP:-1}"
plot_interval_step="${PLOT_INTERVAL_STEP:-1}"
plot_end_step="${PLOT_END_STEP:-500}"

dt_settings=(
    "0.15 5"
    "0.18 6"
    "0.21 7"
    "0.24 8"
    "0.27 9"
    "0.30 10"
    "0.33 11"
    "0.36 12"
    "0.39 13"
    "0.42 14"
    "0.45 15"
)
ensemble_sizes=(10)

# Default to all Lorenz63 observation functions present in the saved runs.
# Override with e.g. OBS_FNS="square arctan" to evaluate only a subset.
if [[ -n "${OBS_FNS:-}" ]]; then
    read -r -a obs_fn_list <<< "$OBS_FNS"
else
    obs_fn_list=(identity arctan square square_root)
fi

contains_obs_fn() {
    local candidate="$1"
    local allowed
    for allowed in "${obs_fn_list[@]}"; do
        if [ "$allowed" = "$candidate" ]; then
            return 0
        fi
    done
    return 1
}

experiments=()
while IFS= read -r cp_path; do
    run_dir="$(dirname "$cp_path")"
    run_name="$(basename "$run_dir")"

    if [[ "$run_name" == *"EtE-LRes"* ]]; then
        v="EtE-LRes"
    elif [[ "$run_name" == *"CorrTerms"* ]]; then
        v="CorrTerms"
    else
        echo "Skipping unknown method directory: $run_name"
        continue
    fi

    if [[ "$run_name" == *"None_"* ]]; then
        obs_fn="${run_name##*None_}"
    else
        echo "Skipping directory with unrecognized obs_fn pattern: $run_name"
        continue
    fi

    if ! contains_obs_fn "$obs_fn"; then
        echo "Skipping obs_fn not in allowlist: $run_name"
        continue
    fi

    experiments+=("$v $cp_path adaptive $obs_fn")
done < <(find "$results_root" -mindepth 2 -maxdepth 2 -type f -name 'cp_1000.pth' | sort)

if [ "${#experiments[@]}" -eq 0 ]; then
    echo "No experiments found under $results_root"
    exit 1
fi

echo "Date: $(date)"
echo "Python: $PYTHON_BIN"
echo "Dataset: $dataset"
echo "Results root: $results_root"
echo "Seed: $seed"
echo "PF_N: $pf_N"
echo "Save Test Figures: $save_test_figures"
echo "Obs Fn allowlist: ${obs_fn_list[*]}"
echo "Ensemble sizes: ${ensemble_sizes[*]}"
echo "dt settings: ${dt_settings[*]}"
echo "Experiments found: ${#experiments[@]}"
echo "----------------------------------------------------"

for exp in "${experiments[@]}"; do
    read -r v cp_path current_sigma_y current_obs_fn <<< "$exp"

    for dt_setting in "${dt_settings[@]}"; do
        read -r dt dt_iter <<< "$dt_setting"
        output_suffix="_dt${dt}"

        echo "=================================================="
        echo "Evaluating Method: $v"
        echo "Checkpoint: $cp_path"
        echo "Sigma Y: $current_sigma_y"
        echo "Obs Fn: $current_obs_fn"
        echo "dt: $dt"
        echo "dt_iter: $dt_iter"
        echo "Output Suffix: $output_suffix"
        echo "=================================================="

        for N in "${ensemble_sizes[@]}"; do
            cmd=(
                "$PYTHON_BIN" evaluate.py
                --dataset "$dataset"
                --N "$N"
                --seed "$seed"
                --dt "$dt"
                --dt_iter "$dt_iter"
                --v "$v"
                --obs_fn "$current_obs_fn"
                --no_localization
                --normal_output
                --test_steps 500
                --pf_verification
                --pf_N "$pf_N"
                --sigma_reg None
                --cp_load_path "$cp_path"
                --suffix "$output_suffix"
            )

            if [[ "$save_test_figures" != "0" && "$save_test_figures" != "false" && "$save_test_figures" != "False" ]]; then
                cmd+=(
                    --save_test_figures
                    --test_snapshot_start_step "$plot_start_step"
                    --test_snapshot_interval "$plot_interval_step"
                    --test_snapshot_end_step "$plot_end_step"
                )
            fi

            if [ "$current_sigma_y" = "adaptive" ]; then
                cmd+=(--adaptive_sigma_y)
            else
                cmd+=(--sigma_y "$current_sigma_y")
            fi

            "${cmd[@]}"
        done
    done
done

echo "All Lorenz63 PF dt evaluations finished."
echo "Completed on $(date)"
