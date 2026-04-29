#!/bin/bash

# Submit this script with: sbatch slurm_scripts/slurm_pf_analysis_cpu.sh

#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=cpu
#SBATCH -J "pf-analysis"
#SBATCH --mail-user=bhchen@caltech.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err

set -euo pipefail

# Change to repo root regardless of the directory used to submit the job.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/bhchen/miniconda3/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    echo "Error: required Python interpreter is not executable: $PYTHON_BIN" >&2
    exit 127
fi

echo "Date: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Host: $(hostname)"
echo "Python: $PYTHON_BIN"
echo "Device: cpu"
echo "----------------------------------------------------"

# Format:
#   "dataset obs_fn sigma_mode test_steps test_traj_num"
# sigma_mode:
#   - "adaptive"          => pass --adaptive_sigma_y
#   - numeric (e.g. 1.0)  => pass --sigma_y <value>
experiments=(
    "lorenz63 square adaptive 500 64"
    "lorenz63 arctan adaptive 500 64"
    "lorenz63 default adaptive 500 64"
    "doubling1d default adaptive 200 64"
)

for exp in "${experiments[@]}"; do
    read -r dataset obs_fn sigma_mode test_steps test_traj_num <<< "$exp"

    echo "============================================================"
    echo "Running PF analysis"
    echo "Dataset: $dataset"
    echo "Obs Fn: $obs_fn"
    echo "Sigma Mode: $sigma_mode"
    echo "Test Steps: $test_steps"
    echo "Test Traj Num: $test_traj_num"
    echo "============================================================"

    cmd=(
        "$PYTHON_BIN" analyze_pf_results.py
        --device "cpu"
        --dataset "$dataset"
        --obs_fn "$obs_fn"
        --test_steps "$test_steps"
        --test_traj_num "$test_traj_num"
        --normal_output
    )

    if [ "$sigma_mode" = "adaptive" ]; then
        cmd+=(--adaptive_sigma_y)
    else
        cmd+=(--sigma_y "$sigma_mode")
    fi

    "${cmd[@]}"
    echo "Done."
    echo ""
done

echo "All PF analysis runs finished."
echo "Job completed on $(date)"
