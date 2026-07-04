#!/usr/bin/env bash
#SBATCH --job-name=loss_lc_sweep
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/loss_lc_sweep_%j.out

# Loss function sweep (MSE / L1 / Hessian-L1 at lc=0.25) and
# lambda_contrastive sweep (nl1 at lc=1/16 and 1/32) for ConAE on vinc.
#
# Baselines for comparison (already trained):
#   nl1_lc025   (lc=0.25)
#   nl1_lc0125  (lc=0.125 = 1/8)

set -eo pipefail
exec 2>&1

REPO="$PWD"
PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
export PYTHONPATH="$REPO"
mkdir -p logs/slurm

CFG="config/contrastive_config"

echo "======================================================================"
echo "Job:   $SLURM_JOB_ID"
echo "Node:  $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Start: $(date)"
echo "======================================================================"

# =============================================================================
# Loss sweep (all lc=0.25)
# =============================================================================
echo ""
echo "### Loss sweep (lc=0.25) ###"

for cfg in \
    "$CFG/ae_contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_mse_lc025.yaml" \
    "$CFG/ae_contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_l1_lc025.yaml" \
    "$CFG/ae_contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_hessian_lc025.yaml"
do
    echo ""
    echo "--- Training: $(basename $cfg .yaml) ---"
    $PYTHON scripts/run_ae_from_config.py "$cfg"
done

# =============================================================================
# Lambda contrastive sweep (nl1, lower lc)
# =============================================================================
echo ""
echo "### Lambda contrastive sweep (nl1) ###"

for cfg in \
    "$CFG/ae_contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_lc0062.yaml" \
    "$CFG/ae_contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_lc0031.yaml"
do
    echo ""
    echo "--- Training: $(basename $cfg .yaml) ---"
    $PYTHON scripts/run_ae_from_config.py "$cfg"
done

echo ""
echo "End: $(date)"
