#!/usr/bin/env bash
#SBATCH --job-name=patchprep_mr10
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/patchprep_mr10_%j.out

# Run CIO-RB patchprep for all 4 datasets × 2 conditions with mask_ratio=0.1.
# Output: ae_results/patches/cio_rb/{dataset}/{condition}/tiff_patches32_mr10/
#
# Dataset  | control | ycomp
# ---------+---------+------
# vinc     |   50    |  41
# pfak     |   10    |   2
# ppax     |   10    |  11
# nih3t3   |   16    |  14

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

for DS in vinc pfak ppax nih3t3; do
    for COND in control ycomp; do
        echo "[$(date)] $DS / $COND ..."
        python scripts/run_patchprep_from_config.py "$CFG/${DS}_${COND}_cio_rb.yaml"
    done
done

echo "[$(date)] ALL DONE"
