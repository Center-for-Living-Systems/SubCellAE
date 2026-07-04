#!/usr/bin/env bash
#SBATCH --job-name=conae_lambda_sweep
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/slurm/conae_lambda_%A_%a.out

# lambda_contrast sweep on lat12proj8: 0.01, 0.05, 0.1
# (baseline is 0.5 — already trained as contrastive_cio_rb_vinc_lat12proj8)

set -eo pipefail
exec 2>&1

LAMBDAS=(lc001 lc005 lc01)
LC=${LAMBDAS[$SLURM_ARRAY_TASK_ID]}

REPO_DIR="$PWD"

source /home/liyading/miniconda3/etc/profile.d/conda.sh
conda activate subcellae-cuda

echo "[$(date)] Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
echo "[$(date)] Starting lambda sweep: ${LC}"
python scripts/run_ae_from_config.py \
  config/contrastive_config/ae_contrastive_cio_rb_vinc_lat12proj8_${LC}.yaml
echo "[$(date)] Done: ${LC}"
