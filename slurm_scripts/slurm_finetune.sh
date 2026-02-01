#!/bin/bash

# Submit this script via submit_finetune.sh (sbatch)

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
# Input: Env vars defined in submit_finetune.sh.
# Output: Fine-tuned model checkpoints and logs.

# 1. Configuration (Defaults for safety)
DATASET=${DATASET:-"ks"}
SEED=${SEED:-42}
EPOCHS=${EPOCHS:-20}
SAVE_EPOCH=${SAVE_EPOCH:-20}
TRAIN_STEPS=${TRAIN_STEPS:-"default"}
TRAIN_TRAJ_NUM=${TRAIN_TRAJ_NUM:-"default"}
LR=${LR:-"default"}
ES_P=${ES_P:-1}
# These must be passed by submit script, but defaults provided to prevent crash
VERSION=${VERSION:-"EtE-LRes"}
CP_PATH=${CP_PATH:-""}
SIGMA_Y=${SIGMA_Y:-1.0}
LOSS_TYPE=${LOSS_TYPE:-"es"}

echo "Starting Finetune: Method=$VERSION, Loss=$LOSS_TYPE, Sig=$SIGMA_Y"
echo "Checkpoint Load: $CP_PATH"

# Load modules
module load cuda/12.2

# Change directory to root
cd ..

# 2. Execution
python finetune.py \
    --epochs $EPOCHS \
    --save_epoch $SAVE_EPOCH \
    --dataset $DATASET \
    --train_steps $TRAIN_STEPS \
    --train_traj_num $TRAIN_TRAJ_NUM \
    --sigma_y $SIGMA_Y \
    --seed $SEED \
    --learning_rate $LR \
    --cp_load_path "$CP_PATH" \
    --v $VERSION \
    --loss_type $LOSS_TYPE \
    --es_p $ES_P \
    --no_running_loss