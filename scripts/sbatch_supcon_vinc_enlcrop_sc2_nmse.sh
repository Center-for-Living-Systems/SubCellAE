#!/usr/bin/env bash
#SBATCH --job-name=supcon_sc2_nmse
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=logs/slurm/supcon_vinc_enlcrop_sc2_nmse_%j.out

# SupCon enlcrop sc2: input/2, normalized MSE, lambda_recon=1.0
# Config: config/contrastive_config/ae_supcon_cio_rb_vinc_lat12proj8_enlcrop_sc2_nmse.yaml

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
    config/contrastive_config/ae_supcon_cio_rb_vinc_lat12proj8_enlcrop_sc2_nmse.yaml

echo ""
echo "End: $(date)"
