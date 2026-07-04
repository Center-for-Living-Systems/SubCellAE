#!/usr/bin/env bash
#SBATCH --job-name=frameextract_vinc
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/frameextract_vinc_%j.out

# Extract full-frame CIO-RB normalized images for vinc control + ycomp.
# Outputs: ae_results/source_frames/cio_rb/vinc/{control,ycomp}/
#   Files: {control,ycomp}_f{NNNN}_pax.tif  (one per source CZI, scale=8.0)

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

echo "[$(date)] Extracting vinc control frames..."
python scripts/run_frameextract_from_config.py "$CFG/vinc_control_cio_rb.yaml"

echo "[$(date)] Extracting vinc ycomp frames..."
python scripts/run_frameextract_from_config.py "$CFG/vinc_ycomp_cio_rb.yaml"

echo "[$(date)] ALL DONE"
