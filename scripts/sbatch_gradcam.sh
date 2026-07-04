#!/usr/bin/env bash
#SBATCH --job-name=gradcam
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --output=logs/slurm/gradcam_%j.out

# GradCAM visualisation: encoder conv features × MLP class logits
# Runs on 4 ppax-trained models (vinc val patches, FA type classification)

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

ROOT="/net/projects/CLS/lding/data/fa_data_analysis"
CRUN="$ROOT/ae_results/contrastive_run"
OUT="$ROOT/ae_results/gradcam"

RUNS=(
    "supcon_cio_rb_vinc_ppax_lat12proj8_enlcrop_sc2_l1"
    "supcon_cio_rb_vinc_ppax_lat12proj8_enlcrop_sc2_nl1_lc025"
    "contrastive_cio_rb_vinc_ppax_lat12proj8_enlcrop_sc2_nl1"
    "contrastive_cio_rb_vinc_ppax_lat12proj8_enlcrop_sc2_nl1_lc025"
)

for run in "${RUNS[@]}"; do
    echo ""
    echo "--- GradCAM: $run ---"
    $PYTHON scripts/run_gradcam.py \
        --ae-dir  "$CRUN/$run" \
        --cls-dir "$CRUN/$run/fa_cls_zrecon_mlp" \
        --out-dir "$OUT/$run" \
        --n-per-class 16
done

echo ""
echo "End: $(date)"
