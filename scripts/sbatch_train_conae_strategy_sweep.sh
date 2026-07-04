#!/usr/bin/env bash
#SBATCH --job-name=conae_strategy_train
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-10
#SBATCH --output=logs/slurm/conae_strategy_train_%A_%a.out

set -eo pipefail
exec 2>&1

STRATEGIES=(0322 0324 mar30 apr08 warmup50 warmup100 0324_nowd mar30_nowd apr08_nowd warmup50_nowd warmup100_nowd)
STRATEGY=${STRATEGIES[$SLURM_ARRAY_TASK_ID]}

REPO_DIR="$PWD"

source /home/liyading/miniconda3/etc/profile.d/conda.sh
conda activate subcellae-cuda

echo "[$(date)] Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
echo "[$(date)] Starting conAE strategy training: ${STRATEGY}"
python scripts/run_ae_from_config.py \
  config/contrastive_config/ae_contrastive_cio_rb_vinc_lat12proj8_${STRATEGY}.yaml
echo "[$(date)] Done: ${STRATEGY}"
