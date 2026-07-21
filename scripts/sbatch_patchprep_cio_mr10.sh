#!/usr/bin/env bash
#SBATCH --job-name=patchprep_cio_mr10
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/patchprep_cio_mr10_%j.out

# Run CIO patchprep (NO rolling ball, scale=1, mask_ratio=0.1) for all 4 datasets × 2 conditions.
# Output: ae_results/patches/cio/{dataset}/{condition}/tiff_patches32_mr10/
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

PYTHON=/home/liyading/miniconda3/bin/python3
export PYTHONPATH="$REPO:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages"

mkdir -p logs/slurm

echo "======================================================================"
echo "[$(date)] Node: $(hostname)"
echo "======================================================================"

for DS in vinc pfak ppax nih3t3; do
    for COND in control ycomp; do
        echo "[$(date)] $DS / $COND ..."
        $PYTHON scripts/run_patchprep_from_config.py "$CFG/${DS}_${COND}_cio.yaml"
    done
done

echo "[$(date)] ALL DONE — CIO mr10 patch prep complete"
