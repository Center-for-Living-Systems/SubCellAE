#!/usr/bin/env bash
#SBATCH --job-name=patchprep_ch3
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=logs/slurm/patchprep_ch3_%j.out

# Extract ch3 (actin) patches for vinc control + ycomp.
# FA positions detected via seg_ch=1 (paxillin); actin intensity extracted at those positions.
# Output: ae_results/patches/cio_rb/vinc_ch3/{control,ycomp}/tiff_patches32_mr10/

set -eo pipefail
exec 2>&1

REPO="$PWD"
CFG="config/patchprep_config"

source /home/liyading/miniconda3/etc/profile.d/conda.sh
conda activate subcellae-cuda

export PYTHONPATH="$REPO"
mkdir -p logs/slurm

echo "======================================================================"
echo "[$(date)] Node: $(hostname)"
echo "======================================================================"

for COND in control ycomp; do
    echo "[$(date)] vinc_ch3 / $COND ..."
    python scripts/run_patchprep_from_config.py "$CFG/vinc_${COND}_cio_rb_ch3.yaml"
done

echo "[$(date)] ALL DONE"
