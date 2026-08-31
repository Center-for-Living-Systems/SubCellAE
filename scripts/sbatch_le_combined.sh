#!/usr/bin/env bash
#SBATCH --job-name=le_comb
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=logs/slurm/le_comb_%A_%a.out

# Combined image-count × label-count efficiency: SupCon AE training array job.
# 225 jobs: 3 series × 15 N_images × 5 npi
# Job list ordered by series (75 jobs per series):
#   Series 0: --array=0-74
#   Series 1: --array=75-149
#   Series 2: --array=150-224
#   All 225 : --array=0-224  (default)

set -eo pipefail
exec 2>&1

REPO="$PWD"
PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
export PYTHONPATH="$REPO"
mkdir -p logs/slurm

JOB_LIST="config/le_combined_job_list.txt"
CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$JOB_LIST")

echo "======================================================================"
echo "Array job : $SLURM_ARRAY_JOB_ID  task $SLURM_ARRAY_TASK_ID"
echo "Config    : $CONFIG"
echo "Node      : $(hostname)"
echo "GPU       : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Start     : $(date)"
echo "======================================================================"

$PYTHON scripts/run_ae_from_config.py "$CONFIG"

echo ""
echo "End: $(date)"
