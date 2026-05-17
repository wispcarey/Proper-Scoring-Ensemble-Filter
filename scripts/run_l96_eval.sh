#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT" || exit 1

PYTHON_BIN="/home/bhchen/miniconda3/bin/python"
dataset="lorenz96"
seed=42
dt=0.15
dt_iter=5
ensemble_sizes=(5 10 15 20 40 60 100)
obs_fn_suffixes=(square_root identity cos2pi square arctan tanh sin cube linear custom)

# Define experiments as pairs: "Results_Subdir Trial_Dirname"
# Format: "lorenz96_results 2026-02-28_14-42lorenz96_1.0_10_60_8192_nl2_joint_CorrTermsNone_identity"
# Previous 0.3 dt / 10 iter experiments live under save/lorenz96_results_0.3.
# experiments=(
#     "lorenz96_results_0.3 2026-02-22_20-32lorenz96_0.27_10_60_8192_es_joint_CorrTermsNone_arctan"
#     "lorenz96_results_0.3 2026-02-22_20-32lorenz96_0.27_10_60_8192_es_joint_EtE-LResNone_arctan"
#     "lorenz96_results_0.3 2026-02-22_20-32lorenz96_0.27_10_60_8192_nl2_joint_CorrTermsNone_arctan"
#     "lorenz96_results_0.3 2026-02-22_20-32lorenz96_0.27_10_60_8192_nl2_joint_EtE-LResNone_arctan"
#     "lorenz96_results_0.3 2026-02-22_20-32lorenz96_6.69_10_60_8192_es_joint_EtE-LResNone_square"
#     "lorenz96_results_0.3 2026-02-22_20-32lorenz96_6.69_10_60_8192_nl2_joint_EtE-LResNone_square"
#     "lorenz96_results_0.3 2026-02-22_20-33lorenz96_6.69_10_60_8192_es_joint_CorrTermsNone_square"
#     "lorenz96_results_0.3 2026-02-22_20-33lorenz96_6.69_10_60_8192_nl2_joint_CorrTermsNone_square"
#     "lorenz96_results_0.3 2026-02-28_14-41lorenz96_1.0_10_60_8192_es_joint_EtE-LResNone_identity"
#     "lorenz96_results_0.3 2026-02-28_14-41lorenz96_1.0_10_60_8192_nl2_joint_EtE-LResNone_identity"
#     "lorenz96_results_0.3 2026-02-28_14-42lorenz96_1.0_10_60_8192_es_joint_CorrTermsNone_identity"
#     "lorenz96_results_0.3 2026-02-28_14-42lorenz96_1.0_10_60_8192_nl2_joint_CorrTermsNone_identity"
#     "lorenz96_results_0.3 2026-05-11_14-12lorenz96_0.27_20_60_8192_es_joint_CorrTerms_tuned_arctan"
#     "lorenz96_results_0.3 2026-05-11_14-12lorenz96_0.27_20_60_8192_es_joint_EtE-LRes_tuned_arctan"
#     "lorenz96_results_0.3 2026-05-11_14-12lorenz96_0.27_20_60_8192_nl2_joint_CorrTerms_tuned_arctan"
#     "lorenz96_results_0.3 2026-05-11_14-29lorenz96_0.27_20_60_8192_nl2_joint_EtE-LRes_tuned_arctan"
#     "lorenz96_results_0.3 2026-05-11_16-04lorenz96_6.69_20_60_8192_es_joint_CorrTerms_tuned_square"
#     "lorenz96_results_0.3 2026-05-11_16-04lorenz96_6.69_20_60_8192_es_joint_EtE-LRes_tuned_square"
#     "lorenz96_results_0.3 2026-05-11_16-04lorenz96_6.69_20_60_8192_nl2_joint_CorrTerms_tuned_square"
#     "lorenz96_results_0.3 2026-05-11_16-04lorenz96_6.69_20_60_8192_nl2_joint_EtE-LRes_tuned_square"
#     "lorenz96_results_0.3 2026-05-11_16-43lorenz96_1.0_20_60_8192_es_joint_EtE-LRes_tuned_identity"
#     "lorenz96_results_0.3 2026-05-11_16-48lorenz96_1.0_20_60_8192_nl2_joint_EtE-LRes_tuned_identity"
#     "lorenz96_results_0.3 2026-05-11_18-41lorenz96_1.0_20_60_8192_es_joint_CorrTerms_tuned_identity"
#     "lorenz96_results_0.3 2026-05-11_18-57lorenz96_1.0_20_60_8192_nl2_joint_CorrTerms_tuned_identity"
# )

experiments=(
    # "lorenz96_results 2026-05-11_20-52lorenz96_0.27_10_60_8192_es_joint_EtE-LResNone_arctan"
    # "lorenz96_results 2026-05-12_11-44lorenz96_0.27_10_60_8192_es_joint_CorrTermsNone_arctan"
    # "lorenz96_results 2026-05-11_20-52lorenz96_0.27_10_60_8192_nl2_joint_EtE-LResNone_arctan"
    # "lorenz96_results 2026-05-11_20-52lorenz96_0.27_10_60_8192_nl2_joint_CorrTermsNone_arctan"
    # "lorenz96_results 2026-05-11_20-52lorenz96_6.69_10_60_8192_es_joint_EtE-LResNone_square"
    # "lorenz96_results 2026-05-11_20-52lorenz96_6.69_10_60_8192_es_joint_CorrTermsNone_square"
    # "lorenz96_results 2026-05-11_20-52lorenz96_6.69_10_60_8192_nl2_joint_EtE-LResNone_square"
    # "lorenz96_results 2026-05-11_20-52lorenz96_6.69_10_60_8192_nl2_joint_CorrTermsNone_square"
    # "lorenz96_results 2026-05-11_21-42lorenz96_1.0_10_60_8192_es_joint_EtE-LResNone_identity"
    # "lorenz96_results 2026-05-11_20-53lorenz96_1.0_10_60_8192_es_joint_CorrTermsNone_identity"
    # "lorenz96_results 2026-05-11_20-52lorenz96_1.0_10_60_8192_nl2_joint_EtE-LResNone_identity"
    # "lorenz96_results 2026-05-12_11-31lorenz96_1.0_10_60_8192_nl2_joint_CorrTermsNone_identity"
    "lorenz96_results 2026-05-14_18-36lorenz96_0.27_20_60_8192_es_joint_EtE-LRes_tuned_arctan"
    "lorenz96_results 2026-05-14_19-16lorenz96_0.27_20_60_8192_es_joint_CorrTerms_tuned_arctan"
    "lorenz96_results 2026-05-14_19-29lorenz96_0.27_20_60_8192_nl2_joint_EtE-LRes_tuned_arctan"
    "lorenz96_results 2026-05-14_19-39lorenz96_0.27_20_60_8192_nl2_joint_CorrTerms_tuned_arctan"
    "lorenz96_results 2026-05-14_17-47lorenz96_6.69_20_60_8192_es_joint_EtE-LRes_tuned_square"
    "lorenz96_results 2026-05-14_17-52lorenz96_6.69_20_60_8192_es_joint_CorrTerms_tuned_square"
    "lorenz96_results 2026-05-14_17-54lorenz96_6.69_20_60_8192_nl2_joint_EtE-LRes_tuned_square"
    "lorenz96_results 2026-05-14_17-54lorenz96_6.69_20_60_8192_nl2_joint_CorrTerms_tuned_square"
    "lorenz96_results 2026-05-14_13-02lorenz96_1.0_20_60_8192_es_joint_EtE-LRes_tuned_identity"
    "lorenz96_results 2026-05-14_13-21lorenz96_1.0_20_60_8192_es_joint_CorrTerms_tuned_identity"
    "lorenz96_results 2026-05-14_16-02lorenz96_1.0_20_60_8192_nl2_joint_EtE-LRes_tuned_identity"
    "lorenz96_results 2026-05-14_17-12lorenz96_1.0_20_60_8192_nl2_joint_CorrTerms_tuned_identity"
)

resolve_results_dir() {
    local results_subdir="$1"
    if [[ "$results_subdir" == save/* ]]; then
        printf '%s\n' "$results_subdir"
    else
        printf 'save/%s\n' "$results_subdir"
    fi
}

infer_method() {
    local trial_name="$1"
    if [[ "$trial_name" == *"CorrTerms"* ]]; then
        printf 'CorrTerms\n'
        return 0
    fi
    if [[ "$trial_name" == *"EtE-LRes"* ]]; then
        printf 'EtE-LRes\n'
        return 0
    fi
    return 1
}

infer_obs_fn() {
    local trial_name="$1"
    local obs_fn
    for obs_fn in "${obs_fn_suffixes[@]}"; do
        if [[ "$trial_name" == *"_${obs_fn}" ]]; then
            printf '%s\n' "$obs_fn"
            return 0
        fi
    done
    printf 'default\n'
}

is_tuned_trial() {
    local trial_name="$1"
    [[ "$trial_name" == *"_tuned"* ]]
}

for exp in "${experiments[@]}"; do
    read -r results_subdir trial_name <<< "$exp"

    results_dir="$(resolve_results_dir "$results_subdir")"
    trial_dir="${results_dir}/${trial_name}"

    if [ ! -d "$results_dir" ]; then
        echo "Skipping missing results directory: $results_dir" >&2
        continue
    fi

    if [ ! -d "$trial_dir" ]; then
        echo "Skipping missing trial directory: $trial_dir" >&2
        continue
    fi

    if ! v="$(infer_method "$trial_name")"; then
        echo "Skipping trial with unknown method: $trial_dir" >&2
        continue
    fi

    current_obs_fn="$(infer_obs_fn "$trial_name")"

    if is_tuned_trial "$trial_name"; then
        checkpoint_mode="finetuned"
    else
        checkpoint_mode="base"
    fi

    echo "=================================================="
    echo "Evaluating Method: $v"
    echo "Results Dir: $results_dir"
    echo "Trial Dir: $trial_dir"
    echo "Checkpoint Mode: $checkpoint_mode"
    echo "Sigma Y: default from config/dataset_info.py"
    echo "dt: $dt"
    echo "dt_iter: $dt_iter"
    echo "Obs Fn: $current_obs_fn"
    echo "=================================================="

    for N in "${ensemble_sizes[@]}"; do
        if [ "$checkpoint_mode" = "finetuned" ]; then
            cp_path="${trial_dir}/ft_cp_${N}_20.pth"
        else
            cp_path="${trial_dir}/cp_1000.pth"
        fi

        if [ ! -f "$cp_path" ]; then
            echo "Skipping missing checkpoint: $cp_path" >&2
            continue
        fi

        cmd=(
            "$PYTHON_BIN" evaluate.py
            --dataset "$dataset"
            --N "$N"
            --seed "$seed"
            --dt "$dt"
            --dt_iter "$dt_iter"
            --v "$v"
            --obs_fn "$current_obs_fn"
            --adaptive_sigma_y
            --normal_output
            --test_steps 1500
            --sigma_reg None
            --cp_load_path "$cp_path"
        )

        "${cmd[@]}"
    done
done
