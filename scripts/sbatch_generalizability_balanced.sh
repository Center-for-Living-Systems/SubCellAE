#!/usr/bin/env bash
#SBATCH --job-name=gen_balanced
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/generalizability_balanced_%j.out

set -eo pipefail
exec 2>&1

REPO_DIR="$PWD"
ROOT="/net/projects/CLS/lding/data/fa_data_analysis"

source /home/liyading/miniconda3/etc/profile.d/conda.sh
conda activate subcellae-cuda

echo "[$(date)] Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"

echo "[$(date)] Training baseline AE: vinc + ppax balanced (ppax oversampled 4x)"
python scripts/run_ae_from_config.py \
  config/generalizability_config/ae_baseline_vinc_ppax_balanced.yaml

echo "[$(date)] Running cross-dataset eval (all 3 models)"
python scripts/run_cross_dataset_eval.py \
  "$ROOT/ae_results/generalizability" \
  --mode sweep \
  --root-folder "$ROOT"

echo "[$(date)] ALL DONE"
