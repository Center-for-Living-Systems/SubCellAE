#!/usr/bin/env bash
#SBATCH --job-name=gen_histmatch
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/generalizability_histmatch_%j.out

# Train baseline AE with histogram matching:
#   1. vinc only         (histmatch)
#   2. vinc + ppax 4x   (histmatch, balanced)
# Then re-run cross-dataset eval across all 5 variants for comparison.

set -eo pipefail
exec 2>&1

REPO_DIR="$PWD"
ROOT="/net/projects/CLS/lding/data/fa_data_analysis"

source /home/liyading/miniconda3/etc/profile.d/conda.sh
conda activate subcellae-cuda

echo "[$(date)] Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"

echo "======================================================================"
echo "[$(date)] Training: vinc only + histogram matching"
echo "======================================================================"
python scripts/run_ae_from_config.py \
  config/generalizability_config/ae_baseline_vinc_only_histmatch.yaml

echo "======================================================================"
echo "[$(date)] Training: vinc + ppax balanced + histogram matching"
echo "======================================================================"
python scripts/run_ae_from_config.py \
  config/generalizability_config/ae_baseline_vinc_ppax_balanced_histmatch.yaml

echo "======================================================================"
echo "[$(date)] Cross-dataset eval (all 5 variants)"
echo "======================================================================"
python scripts/run_cross_dataset_eval.py \
  "$ROOT/ae_results/generalizability" \
  --mode sweep \
  --root-folder "$ROOT"

echo "[$(date)] ALL DONE"
