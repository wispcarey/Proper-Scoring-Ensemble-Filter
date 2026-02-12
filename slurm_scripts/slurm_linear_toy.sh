#!/bin/bash

# Submit this script with: sbatch <this-filename>

#SBATCH --time=1-00:00:00       # walltime (1 day)
#SBATCH --nodes=1               # number of nodes (1 node)
#SBATCH --gres=gpu:1            # 1 GPU
#SBATCH --partition=gpu         # use GPU partition
#SBATCH --ntasks=1              # 1 task
#SBATCH -J "Linear-Train"       # job name
#SBATCH --mail-user=bhchen@caltech.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -o output_record/slurm.%N.%j.out      # STDOUT
#SBATCH -e output_record/slurm.%N.%j.err      # STDERR

# Description: Slurm script for train_v2.py with auto-calculated obs_dim.
# Input: Env vars LOSS_TYPE, EPOCHS, LR, DIM, HIDDEN_DIM, ADAPTIVE_SIGMA_Y, SUFFIX.
# Output: Training logs/models.

# 1. Configuration
LOSS_TYPE=${LOSS_TYPE:-"es"}    # Default loss type
EPOCHS=${EPOCHS:-100}           # Default epochs
LR=${LR:-1e-2}                  # Default learning rate
DIM=${DIM:-10}                  # Default dimension
HIDDEN_DIM=${HIDDEN_DIM:-128}   # Default hidden dimension
SUFFIX=${SUFFIX:-$DIM}          # Default suffix (uses DIM if not set)
ADAPTIVE_SIGMA_Y=${ADAPTIVE_SIGMA_Y:-"false"} # Set to "true" to enable --adaptive_sigma_y

ADAPTIVE_SIGMA_Y_FLAG=""
case "${ADAPTIVE_SIGMA_Y,,}" in
    true|1|yes|y)
        ADAPTIVE_SIGMA_Y_FLAG="--adaptive_sigma_y"
        ;;
esac

# 2. Validation & Calculation
# Check if DIM is even
if (( DIM % 2 != 0 )); then
    echo "Error: DIM ($DIM) must be an even number."
    exit 1
fi

OBS_DIM=$((DIM / 2))
echo "Configuration: DIM=$DIM, OBS_DIM=$OBS_DIM, HIDDEN_DIM=$HIDDEN_DIM, SUFFIX=$SUFFIX, ADAPTIVE_SIGMA_Y=$ADAPTIVE_SIGMA_Y"

# Load modules
module load cuda/12.2

# Change directory
cd ..

# 3. Execution
python train_v2.py \
    --save_epoch 5 \
    --epochs $EPOCHS \
    --dataset linear \
    --N 10 \
    --seed 42 \
    --v EtE-LRes \
    --no_localization \
    --no_running_loss \
    --loss_type $LOSS_TYPE \
    --es_p 1 \
    --dim $DIM \
    --obs_dim $OBS_DIM \
    --hidden_dim $HIDDEN_DIM \
    --learning_rate $LR \
    $ADAPTIVE_SIGMA_Y_FLAG \
    --suffix $SUFFIX
