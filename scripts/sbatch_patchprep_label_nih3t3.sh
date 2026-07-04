#!/usr/bin/env bash
#SBATCH --job-name=patchprep_label_nih3t3
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/patchprep_label_nih3t3_%j.out

set -eo pipefail
exec 2>&1

REPO="$PWD"
CFG="config/patchprep_config"

export PYTHONPATH="$REPO:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages"
PYTHON=/home/liyading/miniconda3/bin/python3

mkdir -p logs/slurm

echo "======================================================================"
echo "[$(date)] Node: $(hostname)"
echo "======================================================================"

echo "[$(date)] nih3t3 control..."
$PYTHON scripts/run_patchprep_from_config.py "$CFG/nih3t3_control_cio_rb_label.yaml"

echo "[$(date)] nih3t3 ycomp..."
$PYTHON scripts/run_patchprep_from_config.py "$CFG/nih3t3_ycomp_cio_rb_label.yaml"

echo "[$(date)] ALL DONE"
