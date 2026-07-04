#!/usr/bin/env bash
#SBATCH --job-name=supcon_analysis
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/supcon_analysis_%j.out

set -eo pipefail
exec 2>&1

REPO_DIR="$PWD"
CFG="config/contrastive_config"

source /home/liyading/miniconda3/etc/profile.d/conda.sh
conda activate subcellae-cuda

echo "======================================================================"
echo "[$(date)] STAGE 2 — Analysis"
echo "======================================================================"
python scripts/run_analysis_from_config.py \
  $CFG/analysis_supcon_cio_rb_vinc_lat12proj8.yaml

echo "======================================================================"
echo "[$(date)] STAGE 3 — Classification (fa/pos × zrecon/zproj)"
echo "======================================================================"
for target in fa pos; do
  for feat in zrecon zproj; do
    echo "--- ${target} | ${feat} ---"
    python scripts/run_classification_from_config.py \
      $CFG/cls_supcon_cio_rb_vinc_lat12proj8_${target}_${feat}.yaml
  done
done

echo "======================================================================"
echo "[$(date)] STAGE 4 — Visualization (zrecon / zproj)"
echo "======================================================================"
for feat in zrecon zproj; do
  echo "--- vis | ${feat} ---"
  python scripts/run_cross_classification_vis.py \
    $CFG/vis_supcon_cio_rb_vinc_lat12proj8_${feat}.yaml
done

echo "[$(date)] ALL DONE"
