#!/usr/bin/env bash
#SBATCH --job-name=train_pax_baseline
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/train_pax_baseline_%j.out

# Train paxillin (ch1) baseline AE:
#   baseline_vinc_only_pax — standard AE, lat12, cosine LR, no contrastive loss
#   Matches baseline_vinc_only_ch3 in all settings except patch dirs (ch1 instead of ch3)

set -eo pipefail
exec 2>&1

REPO="$PWD"
PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
export PYTHONPATH="$REPO"
mkdir -p logs/slurm

echo "======================================================================"
echo "Job:   $SLURM_JOB_ID"
echo "Node:  $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Start: $(date)"
echo "======================================================================"

$PYTHON scripts/run_ae_from_config.py \
    config/contrastive_config/ae_baseline_cio_rb_vinc_only_pax.yaml

echo ""
echo "End: $(date)"
