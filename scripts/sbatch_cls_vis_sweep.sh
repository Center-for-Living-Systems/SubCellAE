#!/usr/bin/env bash
#SBATCH --job-name=cls_vis_sweep
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/cls_vis_%j.out

set -eo pipefail
exec 2>&1

REPO_DIR="$PWD"

source /home/liyading/miniconda3/etc/profile.d/conda.sh
conda activate subcellae-cuda

AE_CFG="config/training_strategies"
CLS_CFG="config/strategy_sweep"

echo "======================================================================"
echo "[$(date)] STAGE 2 — Classification"
echo "======================================================================"

for strategy in 0322 0324 mar30 apr08 warmup50 warmup100 0324_nowd mar30_nowd apr08_nowd warmup50_nowd warmup100_nowd; do
  echo "--- ${strategy} | FA type  | lat8 ---"
  python scripts/run_classification_from_config.py $CLS_CFG/cls_${strategy}_fa_lat8.yaml

  echo "--- ${strategy} | Position | lat8 ---"
  python scripts/run_classification_from_config.py $CLS_CFG/cls_${strategy}_pos_lat8.yaml

  echo "--- ${strategy} | FA type  | lat8+dist8 ---"
  python scripts/run_classification_from_config.py $CLS_CFG/cls_${strategy}_fa_lat8dist8.yaml

  echo "--- ${strategy} | Position | lat8+dist8 ---"
  python scripts/run_classification_from_config.py $CLS_CFG/cls_${strategy}_pos_lat8dist8.yaml
done

echo "======================================================================"
echo "[$(date)] STAGE 3 — Visualization"
echo "======================================================================"

for strategy in 0322 0324 mar30 apr08 warmup50 warmup100 0324_nowd mar30_nowd apr08_nowd warmup50_nowd warmup100_nowd; do
  echo "--- vis ${strategy} | lat8 ---"
  python scripts/run_cross_classification_vis.py $CLS_CFG/vis_${strategy}_lat8.yaml

  echo "--- vis ${strategy} | lat8+dist8 ---"
  python scripts/run_cross_classification_vis.py $CLS_CFG/vis_${strategy}_lat8dist8.yaml
done

echo "[$(date)] ALL DONE"
