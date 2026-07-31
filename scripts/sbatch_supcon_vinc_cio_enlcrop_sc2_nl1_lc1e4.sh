#!/usr/bin/env bash
#SBATCH --job-name=supcon_cio_lc1e4
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=logs/slurm/supcon_cio_vinc_enlcrop_sc2_nl1_lc1e4_%j.out

# SupCon: vinc CIO (no RB), enlcrop sc2 nl1, lambda_contrast=0.0001 (near-zero contrast)
# Config: config/contrastive_config/ae_supcon_cio_vinc_lat12proj8_enlcrop_sc2_nl1_lc1e4.yaml

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
    config/contrastive_config/ae_supcon_cio_vinc_lat12proj8_enlcrop_sc2_nl1_lc1e4.yaml

echo ""
echo "End: $(date)"
