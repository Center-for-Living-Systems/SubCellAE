#!/usr/bin/env bash
#SBATCH --job-name=cio_norm_pptx
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/cio_norm_pptx_%j.out

# Generate cio_norm_before_after.pptx after frame extraction finishes.
# Run after sbatch_frameextract_cio.sh (job 1021320) and
# sbatch_patchprep_cio_mr10.sh (job 1021321) complete.

set -eo pipefail
exec 2>&1

REPO="$PWD"
PYTHON=/home/liyading/miniconda3/bin/python3
export PYTHONPATH="$REPO:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages"

mkdir -p logs/slurm

echo "[$(date)] Node: $(hostname)"
$PYTHON scripts/make_cio_norm_pptx.py
echo "[$(date)] Done — cio_norm_before_after.pptx"
