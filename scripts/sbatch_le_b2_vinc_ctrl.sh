#!/usr/bin/env bash
#SBATCH --job-name=le_vc
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm/le_vc_%A_%a.out

set -eo pipefail
exec 2>&1

REPO="$PWD"
PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
export PYTHONPATH="$REPO"
mkdir -p logs/slurm

JOB_LIST="config/le_b2_vinc_ctrl/job_list.txt"
CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$JOB_LIST")

echo "task $SLURM_ARRAY_TASK_ID -> $CONFIG"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Start: $(date)"

$PYTHON scripts/run_ae_from_config.py "$CONFIG"

echo "End: $(date)"
