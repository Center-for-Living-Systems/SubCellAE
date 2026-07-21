#!/usr/bin/env bash
#SBATCH --job-name=frameextract_cio
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/frameextract_cio_%j.out

# Extract full-frame CIO-normalized images (all 4 channels, NO rolling ball, scale=1)
# for all 4 datasets × 2 conditions.
#
# Dataset  | control | ycomp
# ---------+---------+------
# vinc     |   50    |  41
# pfak     |   10    |   2
# ppax     |   10    |  11
# nih3t3   |   16    |  14
#
# Output: ae_results/source_frames/cio/{dataset}/{condition}/

set -eo pipefail
exec 2>&1

REPO="$PWD"
CFG="config/frameextract_config"

PYTHON=/home/liyading/miniconda3/bin/python3
export PYTHONPATH="$REPO:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages"

mkdir -p logs/slurm

echo "======================================================================"
echo "[$(date)] Node: $(hostname)"
echo "======================================================================"

echo "[$(date)] vinc control (50 × 4 ch)..."
$PYTHON scripts/run_frameextract_from_config.py "$CFG/vinc_control_cio.yaml"

echo "[$(date)] vinc ycomp (41 × 4 ch)..."
$PYTHON scripts/run_frameextract_from_config.py "$CFG/vinc_ycomp_cio.yaml"

echo "[$(date)] pfak control (10 × 4 ch)..."
$PYTHON scripts/run_frameextract_from_config.py "$CFG/pfak_control_cio.yaml"

echo "[$(date)] pfak ycomp (2 × 4 ch)..."
$PYTHON scripts/run_frameextract_from_config.py "$CFG/pfak_ycomp_cio.yaml"

echo "[$(date)] ppax control (10 × 4 ch)..."
$PYTHON scripts/run_frameextract_from_config.py "$CFG/ppax_control_cio.yaml"

echo "[$(date)] ppax ycomp (11 × 4 ch)..."
$PYTHON scripts/run_frameextract_from_config.py "$CFG/ppax_ycomp_cio.yaml"

echo "[$(date)] nih3t3 control (16 × 4 ch)..."
$PYTHON scripts/run_frameextract_from_config.py "$CFG/nih3t3_control_cio.yaml"

echo "[$(date)] nih3t3 ycomp (14 × 4 ch)..."
$PYTHON scripts/run_frameextract_from_config.py "$CFG/nih3t3_ycomp_cio.yaml"

echo "[$(date)] ALL DONE — CIO frame extraction complete"
