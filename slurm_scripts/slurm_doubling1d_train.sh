#!/bin/bash

# Submit this script with: sbatch <this-filename>

#SBATCH --time=1-00:00:00                # walltime (1 day)
#SBATCH --nodes=1                        # number of nodes (1 node)
#SBATCH --gres=gpu:1                     # 1 GPU
#SBATCH --partition=gpu                  # use GPU partition
#SBATCH --ntasks=1                       # 1 task
#SBATCH --mail-user=bhchen@caltech.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -o output_record/lowdim_train.%N.%j.out  # STDOUT
#SBATCH -e output_record/lowdim_train.%N.%j.err  # STDERR

# Description: Slurm script for train.py with configurable parameters for low-dim datasets.
# Input: Env vars EXP_SETTING, DATASET, EPOCHS, N, BATCH_SIZE, SIGMA_Y, VERSION, LOSS_TYPE,
# LOSS_WEIGHTS, USE_PF, LEARNING_RATE, OBS_FN, WEIGHT_DECAY, ADAPTIVE_SIGMA_Y, SUFFIX.
# Fixed in this script: NUM_LOADER_WORKERS=1, SEED=42, PF_N=1000000, TEST_STEPS=200,
# USE_LOCALIZATION=false, USE_RUNNING_LOSS=false, USE_NORMAL_OUTPUT=false, SAVE_TEST_FIGURES=true.

# 1. Configuration
EXP_SETTING=${EXP_SETTING:-""}
case "$EXP_SETTING" in
    ""|"1d")
        DATASET_DEFAULT="doubling1d"
        EXP_SETTING="1d"
        ;;
    "2d")
        DATASET_DEFAULT="complex2d"
        ;;
    *)
        echo "Error: unsupported EXP_SETTING='$EXP_SETTING'. Use '1d' or '2d'." >&2
        exit 1
        ;;
esac

DATASET=${DATASET:-"$DATASET_DEFAULT"}
EPOCHS=${EPOCHS:-100}
N=${N:-30}
BATCH_SIZE=${BATCH_SIZE:-512}
SIGMA_Y=${SIGMA_Y:-0.1}
VERSION=${VERSION:-"EtE-LRes"}
LOSS_TYPE=${LOSS_TYPE:-"es"}
LOSS_WEIGHTS=${LOSS_WEIGHTS:-"None"}
USE_PF=${USE_PF:-"true"}
LEARNING_RATE=${LEARNING_RATE:-"default"}
OBS_FN=${OBS_FN:-"default"}
WEIGHT_DECAY=${WEIGHT_DECAY:-"0"}
ADAPTIVE_SIGMA_Y=${ADAPTIVE_SIGMA_Y:-"false"}
SUFFIX=${SUFFIX:-""}

if [ "$DATASET" = "doubling1d" ]; then
    EXP_SETTING="1d"
elif [ "$DATASET" = "complex2d" ]; then
    EXP_SETTING="2d"
fi

# Fixed values (not configurable from submit script)
NUM_LOADER_WORKERS=1
SEED=42
PF_N=1000000
TEST_STEPS=200

# 2. Boolean flags
PF_FLAG=""
if [ "$USE_PF" = "true" ]; then
    PF_FLAG="--pf_verification"
fi

LOCALIZATION_FLAG="--no_localization"
RUNNING_LOSS_FLAG="--no_running_loss"
SAVE_TEST_FIGURES_FLAG="--save_test_figures"
ADAPTIVE_SIGMA_Y_FLAG=""
case "${ADAPTIVE_SIGMA_Y,,}" in
    true|1|yes|y)
        ADAPTIVE_SIGMA_Y_FLAG="--adaptive_sigma_y"
        ;;
esac

echo "Config: EXP_SETTING=$EXP_SETTING, DATASET=$DATASET, EPOCHS=$EPOCHS, N=$N, BATCH_SIZE=$BATCH_SIZE, SIGMA_Y=$SIGMA_Y, PF=$USE_PF, PF_N=$PF_N, TEST_STEPS=$TEST_STEPS, LR=$LEARNING_RATE, OBS_FN=$OBS_FN, WD=$WEIGHT_DECAY, Adaptive=$ADAPTIVE_SIGMA_Y, SUFFIX='$SUFFIX'"

# Load modules
module load cuda/12.2

# Change directory to project root
cd ..

# 3. Execution
python train.py \
    --dataset "$DATASET" \
    --num_loader_workers "$NUM_LOADER_WORKERS" \
    --epochs "$EPOCHS" \
    --N "$N" \
    --batch_size "$BATCH_SIZE" \
    --sigma_y "$SIGMA_Y" \
    $ADAPTIVE_SIGMA_Y_FLAG \
    --seed "$SEED" \
    --v "$VERSION" \
    --learning_rate "$LEARNING_RATE" \
    --obs_fn "$OBS_FN" \
    --weight_decay "$WEIGHT_DECAY" \
    --suffix "$SUFFIX" \
    --loss_type "$LOSS_TYPE" \
    --loss_weights "$LOSS_WEIGHTS" \
    --es_p 1 \
    --save_epoch 10 \
    --test_steps "$TEST_STEPS" \
    --pf_N "$PF_N" \
    --sigma_reg None \
    $PF_FLAG \
    $LOCALIZATION_FLAG \
    $RUNNING_LOSS_FLAG \
    $SAVE_TEST_FIGURES_FLAG
