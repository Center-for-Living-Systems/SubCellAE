#!/usr/bin/env bash
#SBATCH --job-name=conae_analysis_sweep
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/conae_analysis_%j.out

set -eo pipefail
exec 2>&1

REPO_DIR="$PWD"

source /home/liyading/miniconda3/etc/profile.d/conda.sh
conda activate subcellae-cuda

CFG="config/contrastive_config"

COMBOS=(lat12proj12 lat16proj8 lat16proj12 lat24proj8 lat24proj12 lat32proj8 lat32proj12)

echo "======================================================================"
echo "[$(date)] STAGE 2 — Analysis (dual UMAP z_recon + z_proj)"
echo "======================================================================"

for combo in "${COMBOS[@]}"; do
  echo "--- analysis ${combo} ---"
  python scripts/run_analysis_from_config.py \
    $CFG/analysis_contrastive_cio_rb_vinc_${combo}.yaml
done

echo "======================================================================"
echo "[$(date)] STAGE 3 — Classification (fa/pos × zrecon/zproj)"
echo "======================================================================"

for combo in "${COMBOS[@]}"; do
  for target in fa pos; do
    for feat in zrecon zproj; do
      echo "--- ${combo} | ${target} | ${feat} ---"
      python scripts/run_classification_from_config.py \
        $CFG/cls_contrastive_cio_rb_vinc_${combo}_${target}_${feat}.yaml
    done
  done
done

echo "======================================================================"
echo "[$(date)] STAGE 4 — Visualization (zrecon / zproj)"
echo "======================================================================"

for combo in "${COMBOS[@]}"; do
  for feat in zrecon zproj; do
    echo "--- vis ${combo} | ${feat} ---"
    python scripts/run_cross_classification_vis.py \
      $CFG/vis_contrastive_cio_rb_vinc_${combo}_${feat}.yaml
  done
done

echo "[$(date)] ALL DONE"
