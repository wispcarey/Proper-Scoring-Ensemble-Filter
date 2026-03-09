#!/bin/bash

# Submit this script with: sbatch <this-filename>

#SBATCH --time=1-00:00:00     # walltime (1 days)
#SBATCH --nodes=1           # number of nodes (1 node)
#SBATCH --gres=gpu:p100:1        # 4 GPUs of any type
#SBATCH --partition=gpu     # use GPU partition
#SBATCH --ntasks=1          # 1 task
#SBATCH -J "L63-FT-EtE-LRes-NST-gpu"   # job name
#SBATCH --mail-user=bhchen@caltech.edu # email address
#SBATCH --mail-type=BEGIN   # email notification at start
#SBATCH --mail-type=END     # email notification at end
#SBATCH --mail-type=FAIL    # email notification on failure

# Optional: specify output and error files
#SBATCH -o slurm.%N.%j.out  # STDOUT
#SBATCH -e slurm.%N.%j.err  # STDERR

# Load modules if necessary (e.g., CUDA or other dependencies)
module load cuda/12.2  # Adjusted to CUDA version 12.2

# Change to the directory containing v2_run_fine_tuning.sh
cd ..

# Run your program
python finetune.py \
      --epochs 40 \
      --save_epoch 20 \
      --dataset lorenz63 \
      --train_steps 60 \
      --train_traj_num 8192 \
      --sigma_y 1 \
      --seed 42 \
      --v EtE-LRes \
      --learning_rate 1e-4 \
      --cp_load_path save/2025-07-09_19-08lorenz63_1.0_10_60_8192_es_joint_EtE-LRes_nst/cp_1000.pth \
      --no_running_loss \
      --no_localization \
      --noise_st_input \
      --suffix _nst

