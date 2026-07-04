#!/usr/bin/env bash
#SBATCH --job-name=train_2ch_ae
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/train_2ch_ae_%j.out

# Train 3 two-channel AE models on paxillin (ch1) + actin (ch3) stacked patches:
#   1. baseline_vinc_2ch_pax_act        — standard AE, lat12
#   2. contrastive_...nl1_2ch_pax_act   — ConAE, nl1, lc=0.5  (no enlarged crop)
#   3. contrastive_...nl1_lc025_2ch_pax_act — ConAE, nl1, lc=0.25

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

CFG="config/contrastive_config"

CONFIGS=(
    "$CFG/ae_baseline_cio_rb_vinc_2ch_pax_act.yaml"
    "$CFG/ae_contrastive_cio_rb_vinc_lat12proj8_sc2_nl1_2ch_pax_act.yaml"
    "$CFG/ae_contrastive_cio_rb_vinc_lat12proj8_sc2_nl1_lc025_2ch_pax_act.yaml"
)

for cfg in "${CONFIGS[@]}"; do
    echo ""
    echo "--- Training: $cfg ---"
    $PYTHON scripts/run_ae_from_config.py "$cfg"
done

echo ""
echo "End: $(date)"
