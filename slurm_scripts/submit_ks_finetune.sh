#!/bin/bash

# Description: Batch submit Slurm jobs for finetune.py.
# Input: List of (Method, Checkpoint, Sigma, LossType).
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
LEARNING_RATE="default"
ES_P=1

# List of experiments
# Format: "MethodName Checkpoint_Path Sigma_Y LossType"
EXPERIMENTS=(
    "EtE-LRes save/2026-01-07_16-42ks_1.0_10_60_8192_es_joint_EtE-LRes/cp_1000.pth 1.0 es"
    "CorrTerms save/2026-01-14_02-10ks_1.0_10_60_8192_es_joint_CorrTerms/cp_1000.pth 1.0 es"
    "CorrTerms save/2026-01-07_17-39ks_1.0_10_60_8192_nl2_joint_CorrTerms/cp_1000.pth 1.0 nl2"
)

# Loop: Iterate through configuration
for exp in "${EXPERIMENTS[@]}"; do
    # Parse experiment string
    read -r v cp_path current_sigma_y loss_type <<< "$exp"

    # Construct Job Name
    # Example: ks-ft-EtE-LRes-es
    JOB_NAME="${DATASET}-ft-${v}-${loss_type}"

    echo "Submitting: $JOB_NAME (Sig=$current_sigma_y, Loss=$loss_type, Method=$v)"

    # Submit job with exported variables
    sbatch -J "$JOB_NAME" \
           --time=20:00:00 \
           --export=ALL,DATASET=$DATASET,SEED=$SEED,EPOCHS=$EPOCHS,SAVE_EPOCH=$SAVE_EPOCH,TRAIN_STEPS=$TRAIN_STEPS,TRAIN_TRAJ_NUM=$TRAIN_TRAJ_NUM,LR=$LEARNING_RATE,ES_P=$ES_P,VERSION=$v,CP_PATH=$cp_path,SIGMA_Y=$current_sigma_y,LOSS_TYPE=$loss_type \
           $SLURM_SCRIPT
done