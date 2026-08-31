#!/usr/bin/env bash
#SBATCH --job-name=le_retry
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/slurm/le_retry_%A_%a.out

set -eo pipefail
exec 2>&1

REPO="$PWD"
PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
export PYTHONPATH="$REPO"
mkdir -p logs/slurm

JOB_LIST="config/label_efficiency/job_list_retry.txt"
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
