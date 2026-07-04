#!/usr/bin/env bash
#SBATCH --job-name=frameextract_all
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/frameextract_all_%j.out

# Extract full-frame CIO-RB normalized images (all 4 channels) for all 4 datasets.
# All source CZIs come from the canonical other_paxillin/ folder.
#
# Dataset  | Cond    | CZIs | ch0
# ---------+---------+------+--------
# vinc     | control |  50  | vinculin
# vinc     | ycomp   |  41  | vinculin
# pfak     | control |  10  | pfak
# pfak     | ycomp   |   2  | pfak
# ppax     | control |  10  | ppax
# ppax     | ycomp   |  11  | ppax
# nih3t3   | control |  16  | vinculin
# nih3t3   | ycomp   |  14  | vinculin
#
# Channels per dataset: ch0=marker, ch1=pax(scale=8), ch2=zyx(scale=5), ch3=act(scale=5)
# Output: ae_results/source_frames/cio_rb/{dataset}/{condition}/

set -eo pipefail
exec 2>&1

REPO="$PWD"
CFG="config/frameextract_config"

source /home/liyading/miniconda3/etc/profile.d/conda.sh
conda activate subcellae-cuda

export PYTHONPATH="$REPO"

mkdir -p logs/slurm

echo "======================================================================"
echo "[$(date)] Node: $(hostname)"
echo "======================================================================"

echo "[$(date)] vinc control (50 × 4 ch)..."
python scripts/run_frameextract_from_config.py "$CFG/vinc_control_cio_rb.yaml"

echo "[$(date)] vinc ycomp (41 × 4 ch)..."
python scripts/run_frameextract_from_config.py "$CFG/vinc_ycomp_cio_rb.yaml"

echo "[$(date)] pfak control (10 × 4 ch)..."
python scripts/run_frameextract_from_config.py "$CFG/pfak_control_cio_rb.yaml"

echo "[$(date)] pfak ycomp (2 × 4 ch)..."
python scripts/run_frameextract_from_config.py "$CFG/pfak_ycomp_cio_rb.yaml"

echo "[$(date)] ppax control (10 × 4 ch)..."
python scripts/run_frameextract_from_config.py "$CFG/ppax_control_cio_rb.yaml"

echo "[$(date)] ppax ycomp (11 × 4 ch)..."
python scripts/run_frameextract_from_config.py "$CFG/ppax_ycomp_cio_rb.yaml"

echo "[$(date)] nih3t3 control (16 × 4 ch)..."
python scripts/run_frameextract_from_config.py "$CFG/nih3t3_control_cio_rb.yaml"

echo "[$(date)] nih3t3 ycomp (14 × 4 ch)..."
python scripts/run_frameextract_from_config.py "$CFG/nih3t3_ycomp_cio_rb.yaml"

echo "[$(date)] ALL DONE"
