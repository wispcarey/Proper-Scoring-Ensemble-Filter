#!/bin/bash

# Description: Batch submit Slurm jobs for finetune.py.
# Input: List of
# (Dataset, Method, Checkpoint, Sigma, LossType, LearningRate, ObsFn, AdaptiveSigmaY[true/false]).
# Output: Submits jobs via sbatch.

# Target Slurm script
SLURM_SCRIPT="slurm_finetune.sh"
GPU_TYPE="${1:-p100}"

# Global Configurations
DEF_DATASET="lorenz96"
SEED=42
EPOCHS=20
SAVE_EPOCH=20
TRAIN_STEPS="default"
TRAIN_TRAJ_NUM="default"
DEF_LEARNING_RATE="default"
DEF_OBS_FN="default"
ES_P=1
NORMAL_OUTPUT="true"

# List of experiments
# Format:
# "Dataset MethodName Checkpoint_Path Sigma_Y LossType LearningRate ObsFn AdaptiveSigmaY"
EXPERIMENTS=(
    "lorenz96 CorrTerms save/2026-02-22_20-32lorenz96_0.27_10_60_8192_es_joint_CorrTermsNone_arctan/cp_1000.pth default es default arctan true"
    "lorenz96 EtE-LRes save/2026-02-22_20-32lorenz96_0.27_10_60_8192_es_joint_EtE-LResNone_arctan/cp_1000.pth default es default arctan true"
    "lorenz96 CorrTerms save/2026-02-22_20-32lorenz96_0.27_10_60_8192_nl2_joint_CorrTermsNone_arctan/cp_1000.pth default nl2 default arctan true"
    "lorenz96 EtE-LRes save/2026-02-22_20-32lorenz96_0.27_10_60_8192_nl2_joint_EtE-LResNone_arctan/cp_1000.pth default nl2 default arctan true"
    "lorenz96 CorrTerms save/2026-02-22_20-32lorenz96_0.43_10_60_8192_es_joint_CorrTermsNone_square_root/cp_1000.pth default es default square_root true"
    "lorenz96 EtE-LRes save/2026-02-22_20-32lorenz96_0.43_10_60_8192_es_joint_EtE-LResNone_square_root/cp_1000.pth default es default square_root true"
    "lorenz96 CorrTerms save/2026-02-22_20-32lorenz96_0.43_10_60_8192_nl2_joint_CorrTermsNone_square_root/cp_1000.pth default nl2 default square_root true"
    "lorenz96 EtE-LRes save/2026-02-22_20-32lorenz96_0.43_10_60_8192_nl2_joint_EtE-LResNone_square_root/cp_1000.pth default nl2 default square_root true"
    "lorenz96 EtE-LRes save/2026-02-22_20-32lorenz96_6.69_10_60_8192_es_joint_EtE-LResNone_square/cp_1000.pth default es default square true"
    "lorenz96 EtE-LRes save/2026-02-22_20-32lorenz96_6.69_10_60_8192_nl2_joint_EtE-LResNone_square/cp_1000.pth default nl2 default square true"
    "lorenz96 CorrTerms save/2026-02-22_20-33lorenz96_6.69_10_60_8192_es_joint_CorrTermsNone_square/cp_1000.pth default es default square true"
    "lorenz96 CorrTerms save/2026-02-22_20-33lorenz96_6.69_10_60_8192_nl2_joint_CorrTermsNone_square/cp_1000.pth default nl2 default square true"
    "lorenz96 EtE-LRes save/2026-02-28_14-41lorenz96_1.0_10_60_8192_es_joint_EtE-LResNone_identity/cp_1000.pth default es default default true"
    "lorenz96 EtE-LRes save/2026-02-28_14-41lorenz96_1.0_10_60_8192_nl2_joint_EtE-LResNone_identity/cp_1000.pth default nl2 default default true"
    "lorenz96 CorrTerms save/2026-02-28_14-42lorenz96_1.0_10_60_8192_es_joint_CorrTermsNone_identity/cp_1000.pth default es default default true"
    "lorenz96 CorrTerms save/2026-02-28_14-42lorenz96_1.0_10_60_8192_nl2_joint_CorrTermsNone_identity/cp_1000.pth default nl2 default default true"
)

validate_adaptive_sigma_y() {
    case "${1,,}" in
        true|false|1|0|yes|no|y|n) return 0 ;;
        *)
            echo "Error: ADAPTIVE_SIGMA_Y must be a boolean-like value. Got '$1'" >&2
            return 1
            ;;
    esac
}

dataset_requires_no_localization() {
    case "${1,,}" in
        lorenz63|doubling1d|complex2d) return 0 ;;
        *) return 1 ;;
    esac
}

# Loop: Iterate through configuration
for exp in "${EXPERIMENTS[@]}"; do
    # Parse experiment string
    read -r dataset v cp_path current_sigma_y loss_type learning_rate current_obs_fn adaptive_sigma_y <<< "$exp"
    current_dataset=${dataset:-$DEF_DATASET}
    current_lr=${learning_rate:-$DEF_LEARNING_RATE}
    current_obs_fn=${current_obs_fn:-$DEF_OBS_FN}
    adaptive_sigma_y=${adaptive_sigma_y:-false}

    if ! validate_adaptive_sigma_y "$adaptive_sigma_y"; then
        echo "Malformed experiment entry: $exp" >&2
        exit 1
    fi

    no_localization="false"
    if dataset_requires_no_localization "$current_dataset"; then
        no_localization="true"
    fi

    # Construct Job Name
    # Example: ks-ft-EtE-LRes-es or lorenz96-ft-CorrTerms-nl2-square
    JOB_NAME="${current_dataset}-ft-${v}-${loss_type}"
    if [ "$current_obs_fn" != "default" ] && [ -n "$current_obs_fn" ]; then
        JOB_NAME="${JOB_NAME}-${current_obs_fn}"
    fi

    echo "Submitting: $JOB_NAME (Dataset=$current_dataset, Sig=$current_sigma_y, Loss=$loss_type, Method=$v, LR=$current_lr, ObsFn=$current_obs_fn, Adaptive=$adaptive_sigma_y, NoLocalization=$no_localization)"

    # Submit job with exported variables
    sbatch -J "$JOB_NAME" \
           --time=20:00:00 \
           --gres="gpu:${GPU_TYPE}:1" \
           --export=ALL,DATASET=$current_dataset,SEED=$SEED,EPOCHS=$EPOCHS,SAVE_EPOCH=$SAVE_EPOCH,TRAIN_STEPS=$TRAIN_STEPS,TRAIN_TRAJ_NUM=$TRAIN_TRAJ_NUM,LR=$current_lr,ES_P=$ES_P,VERSION=$v,CP_PATH=$cp_path,SIGMA_Y=$current_sigma_y,LOSS_TYPE=$loss_type,OBS_FN=$current_obs_fn,ADAPTIVE_SIGMA_Y=$adaptive_sigma_y,NORMAL_OUTPUT=$NORMAL_OUTPUT,NO_LOCALIZATION=$no_localization \
           $SLURM_SCRIPT
done
