#!/usr/bin/env bash
#SBATCH --job-name=annabel_vinc_cls
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --array=0-17
#SBATCH --output=logs/slurm/annabel_vinc_cls_%A_%a.out

# Annabel vinc control sweep — 18 classification jobs
# 9 models × (z_recon / z_proj)
# Order: model × split × feat  (conae→supcon2→supcon5, s1v3→s2v2→s3v1, zrecon→zproj)

set -eo pipefail
exec 2>&1

REPO="$PWD"
PYTHON=/home/liyading/miniconda3/bin/python3
export PYTHONPATH="$REPO:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages"

mkdir -p logs/slurm

CONFIGS=($(cat config/annabel_sweep/cls_configs.txt))
CFG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

echo "======================================================================"
echo "Job array : $SLURM_ARRAY_JOB_ID  Task: $SLURM_ARRAY_TASK_ID"
echo "Config    : $CFG"
echo "[$(date)] Node: $(hostname)"
echo "======================================================================"

$PYTHON scripts/run_classification_from_config.py "$CFG"
echo "[$(date)] Done"
