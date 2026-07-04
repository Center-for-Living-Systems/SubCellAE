#!/usr/bin/env bash
#SBATCH --job-name=conae_strategy_analysis
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/conae_strategy_analysis_%j.out

set -eo pipefail
exec 2>&1

REPO_DIR="$PWD"

source /home/liyading/miniconda3/etc/profile.d/conda.sh
conda activate subcellae-cuda

CFG="config/contrastive_config"
STRATEGIES=(0322 0324 mar30 apr08 warmup50 warmup100 0324_nowd mar30_nowd apr08_nowd warmup50_nowd warmup100_nowd)

echo "======================================================================"
echo "[$(date)] STAGE 2 — Analysis"
echo "======================================================================"

for strat in "${STRATEGIES[@]}"; do
  echo "--- analysis lat12proj8_${strat} ---"
  python scripts/run_analysis_from_config.py \
    $CFG/analysis_contrastive_cio_rb_vinc_lat12proj8_${strat}.yaml
done

echo "======================================================================"
echo "[$(date)] STAGE 3 — Classification (fa/pos × zrecon/zproj)"
echo "======================================================================"

for strat in "${STRATEGIES[@]}"; do
  for target in fa pos; do
    for feat in zrecon zproj; do
      echo "--- lat12proj8_${strat} | ${target} | ${feat} ---"
      python scripts/run_classification_from_config.py \
        $CFG/cls_contrastive_cio_rb_vinc_lat12proj8_${strat}_${target}_${feat}.yaml
    done
  done
done

echo "======================================================================"
echo "[$(date)] STAGE 4 — Visualization (zrecon / zproj)"
echo "======================================================================"

for strat in "${STRATEGIES[@]}"; do
  for feat in zrecon zproj; do
    echo "--- vis lat12proj8_${strat} | ${feat} ---"
    python scripts/run_cross_classification_vis.py \
      $CFG/vis_contrastive_cio_rb_vinc_lat12proj8_${strat}_${feat}.yaml
  done
done

echo "[$(date)] ALL DONE"
