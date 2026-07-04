#!/usr/bin/env bash
#SBATCH --job-name=supcon_baseline
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/supcon_baseline_%j.out

set -eo pipefail
exec 2>&1

REPO_DIR="$PWD"

source /home/liyading/miniconda3/etc/profile.d/conda.sh
conda activate subcellae-cuda

echo "[$(date)] Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
echo "[$(date)] Starting SupCon baseline training"
python scripts/run_ae_from_config.py \
  config/contrastive_config/ae_supcon_cio_rb_vinc_lat12proj8.yaml
echo "[$(date)] Training done — submitting analysis job"
sbatch --chdir="$REPO_DIR" \
  "$REPO_DIR/scripts/sbatch_analysis_supcon.sh"
