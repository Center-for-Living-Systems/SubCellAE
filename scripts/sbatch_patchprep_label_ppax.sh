#!/usr/bin/env bash
#SBATCH --job-name=patchprep_label_ppax
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/patchprep_label_ppax_%j.out

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

echo "[$(date)] ppax control..."
$PYTHON scripts/run_patchprep_from_config.py "$CFG/ppax_control_cio_rb_label.yaml"

echo "[$(date)] ppax ycomp..."
$PYTHON scripts/run_patchprep_from_config.py "$CFG/ppax_ycomp_cio_rb_label.yaml"

echo "[$(date)] ALL DONE"
