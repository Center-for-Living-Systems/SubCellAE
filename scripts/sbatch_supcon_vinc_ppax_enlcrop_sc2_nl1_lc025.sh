#!/usr/bin/env bash
#SBATCH --job-name=supcon_ppax_nl1_025
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=logs/slurm/supcon_vinc_ppax_enlcrop_sc2_nl1_lc025_%j.out

# SupCon: vinc+ppax balanced (ds1+ds3), enlcrop sc2 nl1, lambda_contrast=0.25, ppax ×4
# Config: config/contrastive_config/ae_supcon_cio_rb_vinc_ppax_lat12proj8_enlcrop_sc2_nl1_lc025.yaml

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
    config/contrastive_config/ae_supcon_cio_rb_vinc_ppax_lat12proj8_enlcrop_sc2_nl1_lc025.yaml

echo ""
echo "End: $(date)"
