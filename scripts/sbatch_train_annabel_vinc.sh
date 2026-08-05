#!/usr/bin/env bash
#SBATCH --job-name=annabel_vinc_train
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --array=0-8
#SBATCH --output=logs/slurm/annabel_vinc_train_%A_%a.out

# Annabel vinc control sweep — 9 training jobs
# (conae / supcon2 / supcon5) × (s1v3 / s2v2 / s3v1)

set -eo pipefail
exec 2>&1

REPO="$PWD"
PYTHON=/home/liyading/miniconda3/bin/python3
export PYTHONPATH="$REPO:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages"

mkdir -p logs/slurm

CONFIGS=($(cat config/annabel_sweep/train_configs.txt))
CFG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

echo "======================================================================"
echo "Job array : $SLURM_ARRAY_JOB_ID  Task: $SLURM_ARRAY_TASK_ID"
echo "Config    : $CFG"
echo "[$(date)] Node: $(hostname)"
echo "======================================================================"

$PYTHON scripts/run_ae_from_config.py "$CFG"
echo "[$(date)] Done"
