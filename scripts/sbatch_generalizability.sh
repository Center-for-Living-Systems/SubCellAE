#!/usr/bin/env bash
#SBATCH --job-name=gen_baseline
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/generalizability_%j.out

# Train baseline AE on vinc-only and vinc+ppax, then run cross-dataset eval.
# Compares reconstruction L1 on train/val/pfak/nih3t3 to test generalizability.

set -eo pipefail
exec 2>&1

REPO_DIR="$PWD"
ROOT="/net/projects/CLS/lding/data/fa_data_analysis"

source /home/liyading/miniconda3/etc/profile.d/conda.sh
conda activate subcellae-cuda

echo "======================================================================"
echo "[$(date)] Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
echo "======================================================================"

echo "[$(date)] Training baseline AE: vinc-only"
python scripts/run_ae_from_config.py \
  config/generalizability_config/ae_baseline_vinc_only.yaml

echo "[$(date)] Training baseline AE: vinc + ppax"
python scripts/run_ae_from_config.py \
  config/generalizability_config/ae_baseline_vinc_ppax.yaml

echo "[$(date)] Running cross-dataset eval"
python scripts/run_cross_dataset_eval.py \
  "$ROOT/ae_results/generalizability" \
  --mode sweep \
  --root-folder "$ROOT"

echo "[$(date)] ALL DONE"
