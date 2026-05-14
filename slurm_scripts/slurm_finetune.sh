#!/bin/bash

# Submit this script via a batch submit wrapper such as submit_highdim_finetune.sh (sbatch)

#SBATCH --time=04:00:00         # walltime (adjust as needed)
#SBATCH --nodes=1               # number of nodes
#SBATCH --gres=gpu:1            # 1 GPU
#SBATCH --partition=gpu         # use GPU partition
#SBATCH --ntasks=1              # 1 task
#SBATCH --mail-user=bhchen@caltech.edu
#SBATCH --mail-type=FAIL
#SBATCH -o output_record/ft.%x.%j.out      # STDOUT (%x=job_name)
#SBATCH -e output_record/ft.%x.%j.err      # STDERR

# Description: Slurm execution script for finetune.py.
# Input: Env vars defined in a submit wrapper (including optional OBS_FN, ADAPTIVE_SIGMA_Y, and DETACH_STEPS).
# Output: Fine-tuned model checkpoints and logs.

# 1. Configuration (Defaults for safety)
DATASET=${DATASET:-"ks"}
SEED=${SEED:-42}
EPOCHS=${EPOCHS:-20}
SAVE_EPOCH=${SAVE_EPOCH:-20}
TRAIN_STEPS=${TRAIN_STEPS:-"default"}
TRAIN_TRAJ_NUM=${TRAIN_TRAJ_NUM:-"default"}
LR=${LR:-"default_ft"}
ES_P=${ES_P:-1}
# These must be passed by submit script, but defaults provided to prevent crash
VERSION=${VERSION:-"EtE-LRes"}
CP_PATH=${CP_PATH:-""}
SIGMA_Y=${SIGMA_Y:-1.0}
LOSS_TYPE=${LOSS_TYPE:-"es"}
OBS_FN=${OBS_FN:-"default"}
ADAPTIVE_SIGMA_Y=${ADAPTIVE_SIGMA_Y:-"false"}
NORMAL_OUTPUT=${NORMAL_OUTPUT:-"false"}
NO_LOCALIZATION=${NO_LOCALIZATION:-"false"}
DETACH_STEPS=${DETACH_STEPS:-5}
PYTHON_BIN=${PYTHON_BIN:-"/home/bhchen/miniconda3/bin/python"}

ADAPTIVE_SIGMA_Y_FLAG=""
case "${ADAPTIVE_SIGMA_Y,,}" in
    true|1|yes|y)
        ADAPTIVE_SIGMA_Y_FLAG="--adaptive_sigma_y"
        ;;
esac

NORMAL_OUTPUT_FLAG=""
case "${NORMAL_OUTPUT,,}" in
    true|1|yes|y)
        NORMAL_OUTPUT_FLAG="--normal_output"
        ;;
esac

NO_LOCALIZATION_FLAG=""
case "${DATASET,,}" in
    lorenz63|doubling1d|complex2d)
        NO_LOCALIZATION_FLAG="--no_localization"
        ;;
esac
case "${NO_LOCALIZATION,,}" in
    true|1|yes|y)
        NO_LOCALIZATION_FLAG="--no_localization"
        ;;
esac

echo "Starting Finetune: Dataset=$DATASET, Method=$VERSION, Loss=$LOSS_TYPE, Sig=$SIGMA_Y, LR=$LR, ObsFn=$OBS_FN, Adaptive=$ADAPTIVE_SIGMA_Y, DetachSteps=$DETACH_STEPS, NormalOutput=$NORMAL_OUTPUT, NoLocalization=$([ -n "$NO_LOCALIZATION_FLAG" ] && echo true || echo false)"
echo "Checkpoint Load: $CP_PATH"

# Load modules
module load cuda/12.2

# Change directory to root
cd ..

# 2. Execution
"$PYTHON_BIN" finetune.py \
    --epochs $EPOCHS \
    --save_epoch $SAVE_EPOCH \
    --dataset $DATASET \
    --train_steps $TRAIN_STEPS \
    --train_traj_num $TRAIN_TRAJ_NUM \
    --sigma_y $SIGMA_Y \
    $ADAPTIVE_SIGMA_Y_FLAG \
    --seed $SEED \
    --learning_rate $LR \
    --cp_load_path "$CP_PATH" \
    --v $VERSION \
    $NO_LOCALIZATION_FLAG \
    $NORMAL_OUTPUT_FLAG \
    --obs_fn $OBS_FN \
    --loss_type $LOSS_TYPE \
    --es_p $ES_P \
    --detach_steps $DETACH_STEPS \
    --no_running_loss
