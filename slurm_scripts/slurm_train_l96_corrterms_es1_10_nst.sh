#!/bin/bash

# Submit this script with: sbatch <this-filename>

#SBATCH --time=1-00:00:00     # walltime (1 days)
#SBATCH --nodes=1           # number of nodes (1 node)
#SBATCH --gres=gpu:1        # 4 GPUs of any type
#SBATCH --partition=gpu     # use GPU partition
#SBATCH --ntasks=1          # 1 task
#SBATCH -J "L96-CorrTerms-nst-gpu"   # job name
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
python train.py \
   --dataset lorenz96 \
   --epochs 1000 \
   --N 10 \
   --sigma_y 1 \
   --seed 42 \
   --v CorrTerms \
   --loss_type es \
   --es_p 1 \
   --test_steps 500 \
   --no_running_loss \
   --noise_st_input \
   --suffix _nst

