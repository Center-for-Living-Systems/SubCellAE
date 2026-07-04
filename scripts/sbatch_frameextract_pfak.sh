#!/usr/bin/env bash
#SBATCH --job-name=frameextract_pfak
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/frameextract_pfak_%j.out

set -eo pipefail
exec 2>&1

REPO="$PWD"
CFG="config/frameextract_config"

export PYTHONPATH="$REPO:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages"
PYTHON=/home/liyading/miniconda3/bin/python3

mkdir -p logs/slurm

echo "======================================================================"
echo "[$(date)] Node: $(hostname)"
echo "======================================================================"

echo "[$(date)] pfak control (10 × 4 ch)..."
$PYTHON scripts/run_frameextract_from_config.py "$CFG/pfak_control_cio_rb.yaml"

echo "[$(date)] pfak ycomp (2 × 4 ch)..."
$PYTHON scripts/run_frameextract_from_config.py "$CFG/pfak_ycomp_cio_rb.yaml"

echo "[$(date)] ALL DONE"
