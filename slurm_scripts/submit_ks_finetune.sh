#!/bin/bash

# Description: Batch submit Slurm jobs for finetune.py.
# Input: List of (Method, Checkpoint, Sigma, LossType, LearningRate, AdaptiveSigmaY[true/false]).
# Output: Submits jobs via sbatch.

# Target Slurm script
SLURM_SCRIPT="slurm_finetune.sh"

# Global Configurations
DATASET="ks"
SEED=42
EPOCHS=20
SAVE_EPOCH=20
TRAIN_STEPS="default"
TRAIN_TRAJ_NUM="default"
DEF_LEARNING_RATE="default"
ES_P=1

# List of experiments
# Format: "MethodName Checkpoint_Path Sigma_Y LossType LearningRate AdaptiveSigmaY"
EXPERIMENTS=(
    "EtE-LRes save/2026-01-07_16-42ks_1.0_10_60_8192_es_joint_EtE-LRes/cp_1000.pth 1.0 es default false"
    "CorrTerms save/2026-01-14_02-10ks_1.0_10_60_8192_es_joint_CorrTerms/cp_1000.pth 1.0 es default false"
    "CorrTerms save/2026-01-07_17-39ks_1.0_10_60_8192_nl2_joint_CorrTerms/cp_1000.pth 1.0 nl2 default false"
)

# Loop: Iterate through configuration
for exp in "${EXPERIMENTS[@]}"; do
    # Parse experiment string
    read -r v cp_path current_sigma_y loss_type learning_rate adaptive_sigma_y <<< "$exp"
    current_lr=${learning_rate:-$DEF_LEARNING_RATE}
    case "$adaptive_sigma_y" in
        true|false) ;;
        *)
            echo "Error: ADAPTIVE_SIGMA_Y must be 'true' or 'false'. Got '$adaptive_sigma_y'" >&2
            echo "Malformed experiment entry: $exp" >&2
            exit 1
            ;;
    esac

    # Construct Job Name
    # Example: ks-ft-EtE-LRes-es
    JOB_NAME="${DATASET}-ft-${v}-${loss_type}"

    echo "Submitting: $JOB_NAME (Sig=$current_sigma_y, Loss=$loss_type, Method=$v, LR=$current_lr, Adaptive=$adaptive_sigma_y)"

    # Submit job with exported variables
    sbatch -J "$JOB_NAME" \
           --time=20:00:00 \
           --export=ALL,DATASET=$DATASET,SEED=$SEED,EPOCHS=$EPOCHS,SAVE_EPOCH=$SAVE_EPOCH,TRAIN_STEPS=$TRAIN_STEPS,TRAIN_TRAJ_NUM=$TRAIN_TRAJ_NUM,LR=$current_lr,ES_P=$ES_P,VERSION=$v,CP_PATH=$cp_path,SIGMA_Y=$current_sigma_y,LOSS_TYPE=$loss_type,ADAPTIVE_SIGMA_Y=$adaptive_sigma_y \
           $SLURM_SCRIPT
done
