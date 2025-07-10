#!/bin/bash

# Submit this script with: sbatch <this-filename>

#SBATCH --time=1-00:00:00     # walltime (1 days)
#SBATCH --nodes=1           # number of nodes (1 node)
#SBATCH --gres=gpu:1        # 1 GPUs of any type
#SBATCH --partition=gpu     # use GPU partition
#SBATCH --ntasks=1          # 1 task
#SBATCH -J "2ddoublewell-explode-test"   # job name
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
    --epochs 1000 \
    --dataset Hdoublewell \
    --N 10 \
    --seed 42 \
    --v CorrTerms \
    --no_localization \
    --loss_type es \
    --es_p 1 \
    --weight_decay 1e-2 
    
python train.py \
    --epochs 1000 \
    --dataset Hdoublewell \
    --N 10 \
    --seed 42 \
    --v CorrTerms \
    --no_localization \
    --loss_type es \
    --es_p 1 \
    --weight_decay 1e-2 \
    --noise_st_input \
    --mlp_y_type noise_innov \
    --suffix _nst_nimlp

python train.py \
    --epochs 1000 \
    --dataset Hdoublewell \
    --N 10 \
    --seed 42 \
    --v CorrTerms \
    --no_localization \
    --loss_type es \
    --es_p 1 \
    --weight_decay 1e-2 \
    --noise_st_input \
    --suffix _nst

python train.py \
    --epochs 1000 \
    --dataset Hdoublewell \
    --N 10 \
    --seed 42 \
    --v CorrTerms \
    --no_localization \
    --loss_type es \
    --es_p 1 \
    --weight_decay 1e-2 \
    --mlp_y_type noise_innov \
    --suffix _nimlp


