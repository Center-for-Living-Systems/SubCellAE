#!/usr/bin/env bash
#SBATCH --job-name=img_eff
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=logs/slurm/img_eff_%A_%a.out

# Image-count efficiency: SupCon AE training array job.
# 45 jobs: 15 N_images × 3 repeats
# Job list: config/img_eff_job_list.txt  (1 config per line)
#
# Series / N grouping (0-indexed tasks):
#   All 45 jobs: --array=0-44  (default)
#
# Submit: sbatch --array=0-44 scripts/sbatch_image_efficiency.sh

set -eo pipefail
exec 2>&1

REPO="$PWD"
PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
export PYTHONPATH="$REPO"
mkdir -p logs/slurm

JOB_LIST="config/img_eff_job_list.txt"
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
