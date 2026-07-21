#!/usr/bin/env bash
#SBATCH --job-name=cio_comparison_pptx
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/cio_comparison_pptx_%j.out

set -eo pipefail
exec 2>&1

REPO="$PWD"
PYTHON=/home/liyading/miniconda3/bin/python3
export PYTHONPATH="$REPO:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages"

mkdir -p logs/slurm

echo "[$(date)] Node: $(hostname)"
$PYTHON scripts/make_cio_comparison_pptx.py
echo "[$(date)] Done — cio_comparison_pax.pptx"
