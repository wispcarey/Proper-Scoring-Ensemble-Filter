#!/bin/bash

# Submit this script with: sbatch <this-filename>

#SBATCH --time=1-00:00:00     # walltime (1 days)
#SBATCH --nodes=1           # number of nodes (1 node)
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu     # use GPU partition
#SBATCH --ntasks=1          # 1 task
#SBATCH -J "bpf-rossler-vis"   # job name
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

echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Python: $(which python)"
echo "----------------------------------------------------"

SEEDS=(42)
PARTICLE_NUMBERS=(100000)

for seed_val in "${SEEDS[@]}"; do
    for pf_n_val in "${PARTICLE_NUMBERS[@]}"; do
        
        echo "============================================================"
        echo "Running with Seed: $seed_val and Particle Count (pf_N): $pf_n_val"
        echo "============================================================"
        
        python gen_pf_results.py \
            --dataset lorenz96 \
            --sigma_y 1 \
            --seed "$seed_val" \
            --normal_output \
            --test_steps 500 \
            --pf_verification \
            --pf_N "$pf_n_val" \
            --sigma_reg None \
            --pf_save_figure
        
        echo "Done."
        echo ""
    done
done

echo "All experiments finished."
echo "Job completed on $(date)"
