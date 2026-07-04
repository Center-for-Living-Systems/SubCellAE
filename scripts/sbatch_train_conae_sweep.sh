#!/usr/bin/env bash
#SBATCH --job-name=conae_train_sweep
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-6
#SBATCH --output=logs/slurm/conae_train_%A_%a.out

set -eo pipefail
exec 2>&1

# baseline (lat12proj8) already done; 7 new combos
COMBOS=(lat12proj12 lat16proj8 lat16proj12 lat24proj8 lat24proj12 lat32proj8 lat32proj12)
COMBO=${COMBOS[$SLURM_ARRAY_TASK_ID]}

REPO_DIR="$PWD"

source /home/liyading/miniconda3/etc/profile.d/conda.sh
conda activate subcellae-cuda

echo "[$(date)] Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
echo "[$(date)] Starting conAE training: ${COMBO}"
python scripts/run_ae_from_config.py \
  config/contrastive_config/ae_contrastive_cio_rb_vinc_${COMBO}.yaml
echo "[$(date)] Done: ${COMBO}"
