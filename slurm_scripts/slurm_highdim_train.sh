#!/bin/bash

# Submit this script with: sbatch <this-filename>

#SBATCH --time=1-00:00:00               # walltime (1 day)
#SBATCH --nodes=1                       # number of nodes (1 node)
#SBATCH --gres=gpu:1                    # 1 GPU
#SBATCH --partition=gpu                 # use GPU partition
#SBATCH --exclude=hpc-93-36             # avoid this node; any other node is allowed
#SBATCH --ntasks=1                      # 1 task
#SBATCH --mail-user=bhchen@caltech.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -o output_record/highdim_train.%N.%j.out      # STDOUT (saved to output_record)
#SBATCH -e output_record/highdim_train.%N.%j.err      # STDERR (saved to output_record)

# Description: Slurm script for train.py with configurable parameters for Lorenz63.
# Input: Env vars DATASET, EPOCHS, N, SIGMA_Y, VERSION, LOSS_TYPE, LOSS_WEIGHTS, LEARNING_RATE, OBS_FN, WEIGHT_DECAY, ADAPTIVE_SIGMA_Y, SUFFIX, DETACH_STEPS.
# Output: Training logs/models.

# 1. Configuration
DATASET=${DATASET:-"lorenz63"}       # Default dataset
EPOCHS=${EPOCHS:-1000}               # Default epochs
N=${N:-10}                           # Default N
SIGMA_Y=${SIGMA_Y:-1}                # Default sigma_y
VERSION=${VERSION:-"EtE-LRes"}       # Default version (--v)
LOSS_TYPE=${LOSS_TYPE:-"es"}         # Default loss type
LOSS_WEIGHTS=${LOSS_WEIGHTS:-"none"} # Default loss weights
LEARNING_RATE=${LEARNING_RATE:-"default"} # Default learning rate
OBS_FN=${OBS_FN:-"identity"}         # Default observation post-function
WEIGHT_DECAY=${WEIGHT_DECAY:-"0"}    # Default weight decay
ADAPTIVE_SIGMA_Y=${ADAPTIVE_SIGMA_Y:-"false"} # Set to "true" to enable --adaptive_sigma_y
SUFFIX=${SUFFIX:-""}                 # Default suffix (empty string)
DETACH_STEPS=${DETACH_STEPS:-5}      # Default detach interval

ADAPTIVE_SIGMA_Y_FLAG=""
case "${ADAPTIVE_SIGMA_Y,,}" in
    true|1|yes|y)
        ADAPTIVE_SIGMA_Y_FLAG="--adaptive_sigma_y"
        ;;
esac

echo "Config: DATASET=$DATASET, EPOCHS=$EPOCHS, N=$N, v=$VERSION, LR=$LEARNING_RATE, obs_fn=$OBS_FN, weight_decay=$WEIGHT_DECAY, adaptive_sigma_y=$ADAPTIVE_SIGMA_Y, SUFFIX='$SUFFIX', detach_steps=$DETACH_STEPS"

# Load modules
module load cuda/12.2

# Change directory
cd ..

# 3. Execution
python train.py \
    --dataset $DATASET \
    --num_loader_workers 1 \
    --epochs $EPOCHS \
    --N $N \
    --sigma_y $SIGMA_Y \
    $ADAPTIVE_SIGMA_Y_FLAG \
    --seed 42 \
    --v $VERSION \
    --learning_rate $LEARNING_RATE \
    --obs_fn $OBS_FN \
    --weight_decay $WEIGHT_DECAY \
    --suffix "$SUFFIX" \
    --detach_steps $DETACH_STEPS \
    --loss_type $LOSS_TYPE \
    --loss_weights $LOSS_WEIGHTS \
    --es_p 1 \
    --save_epoch 10
