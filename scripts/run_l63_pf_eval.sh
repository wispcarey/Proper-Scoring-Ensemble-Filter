#!/bin/bash

cd ..

dataset="lorenz63"
seed=42

# Define experiments as quartets: "MethodName Checkpoint_Path Sigma_Y Obs_Fn"
# Format: "MethodName Path/To/Checkpoint SigmaValue ObsFn"
results_root="save/lorenz63_results"
obs_fn_list=("identity")

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

# Loop 1: Iterate through configuration
for exp in "${experiments[@]}"; do
    # Extract version, path, sigma_y, and obs_fn
    read -r v cp_path current_sigma_y current_obs_fn <<< "$exp"
    
    echo "=================================================="
    echo "Evaluating Method: $v"
    echo "Checkpoint: $cp_path"
    echo "Sigma Y: $current_sigma_y"
    echo "Obs Fn: $current_obs_fn"
    echo "=================================================="

    # Loop 2: Iterate through N
    for N in 5 10 15 20 40 60 100; do
        cmd=(
            python evaluate.py
            --dataset "$dataset"
            --N "$N"
            --seed "$seed"
            --v "$v"
            --obs_fn "$current_obs_fn"
            --no_localization
            --normal_output
            --test_steps 500
            --pf_verification
            --pf_N 1000000
            --sigma_reg None
            --cp_load_path "$cp_path"
            --save_test_figures
        )

        if [ "$current_sigma_y" = "adaptive" ]; then
            cmd+=(--adaptive_sigma_y)
        else
            cmd+=(--sigma_y "$current_sigma_y")
        fi

        "${cmd[@]}"
    done
done
