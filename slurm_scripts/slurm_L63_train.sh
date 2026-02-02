#!/bin/bash

# Submit this script with: sbatch <this-filename>

#SBATCH --time=1-00:00:00        # walltime (1 day)
#SBATCH --nodes=1                # number of nodes (1 node)
#SBATCH --gres=gpu:1             # 1 GPU
#SBATCH --partition=gpu          # use GPU partition
#SBATCH --ntasks=1               # 1 task
#SBATCH --mail-user=bhchen@caltech.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -o output_record/L63_train.%N.%j.out      # STDOUT (saved to output_record)
#SBATCH -e output_record/L63_train.%N.%j.err      # STDERR (saved to output_record)

# Description: Slurm script for train.py with configurable parameters for Lorenz63.
# Input: Env vars DATASET, EPOCHS, N, SIGMA_Y, VERSION, LOSS_TYPE, LOSS_WEIGHTS, USE_PF, LEARNING_RATE, SUFFIX.
# Output: Training logs/models.

# 1. Configuration
DATASET=${DATASET:-"lorenz63"}       # Default dataset
EPOCHS=${EPOCHS:-1000}               # Default epochs
N=${N:-10}                           # Default N
SIGMA_Y=${SIGMA_Y:-1}                # Default sigma_y
VERSION=${VERSION:-"EtE-LRes"}       # Default version (--v)
LOSS_TYPE=${LOSS_TYPE:-"es"}         # Default loss type
LOSS_WEIGHTS=${LOSS_WEIGHTS:-"none"} # Default loss weights
USE_PF=${USE_PF:-"true"}             # Set to "false" to disable --pf_verification
LEARNING_RATE=${LEARNING_RATE:-"default"} # Default learning rate
SUFFIX=${SUFFIX:-""}                 # Default suffix (empty string)

# 2. Logic for Boolean Flags
PF_FLAG=""
if [ "$USE_PF" = "true" ]; then
    PF_FLAG="--pf_verification"
fi

echo "Config: DATASET=$DATASET, EPOCHS=$EPOCHS, N=$N, v=$VERSION, PF=$USE_PF, LR=$LEARNING_RATE, SUFFIX='$SUFFIX', LOSS_WEIGHTS=$LOSS_WEIGHTS"

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
    --seed 42 \
    --v $VERSION \
    --learning_rate $LEARNING_RATE \
    --suffix "$SUFFIX" \
    --no_localization \
    --no_running_loss \
    --loss_type $LOSS_TYPE \
    --loss_weights $LOSS_WEIGHTS \
    --es_p 1 \
    --test_steps 500 \
    $PF_FLAG \
    --pf_N 1000000 \
    --sigma_reg None \
    --save_epoch 10