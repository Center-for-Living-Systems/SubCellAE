#!/usr/bin/env bash
#SBATCH --job-name=patchprep_label
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/patchprep_label_%j.out

# Patchprep for labelling — all 4 datasets × 2 conditions.
# Changes vs training configs:
#   seg_threshold : 0.05  (was 0.1) — wider cell mask, more cell area covered
#   mask_ratio    : 0.1   (was 0.4) — patch needs only 10% cell coverage
# Output: ae_results/patches/cio_rb/{dataset}/{condition}/tiff_patches32_label/

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
        python scripts/run_patchprep_from_config.py "$CFG/${DS}_${COND}_cio_rb_label.yaml"
    done
done

echo "[$(date)] ALL DONE"
