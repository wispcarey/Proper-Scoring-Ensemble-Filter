#!/bin/bash

# Submit this script with: sbatch slurm_scripts/slurm_pf_analysis_cpu.sh

#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -J "pf-analysis"
#SBATCH --mail-user=bhchen@caltech.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -o pf_analysis/slurm.%N.%j.out
#SBATCH -e pf_analysis/slurm.%N.%j.err
#SBATCH --chdir=/home/bhchen/LearnKalmanGain

set -euo pipefail

# Slurm may execute a copied script from /var/spool/slurmd, so do not infer the
# repository path from BASH_SOURCE. Override REPO_ROOT via sbatch --export if needed.
REPO_ROOT="${REPO_ROOT:-/home/bhchen/LearnKalmanGain}"
if [ ! -f "$REPO_ROOT/analyze_pf_results.py" ]; then
    echo "Error: REPO_ROOT='$REPO_ROOT' does not contain analyze_pf_results.py." >&2
    exit 1
fi
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: Python command is not available in PATH: $PYTHON_BIN" >&2
    exit 127
fi

echo "Date: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Host: $(hostname)"
echo "Python: $(command -v "$PYTHON_BIN")"
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
