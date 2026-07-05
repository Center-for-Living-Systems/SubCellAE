#!/usr/bin/env bash
#SBATCH --job-name=ds_combo_eval
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/ds_combo_eval_%j.out

# Post-training eval + analysis for all 15 dataset-combo models.
# For each model:
#   1. Cross-dataset reconstruction eval on all 4 ds (violin plots)
#   2. UMAP + KMeans cluster panels on z_proj latents
#
# Designed to run after sbatch_train_ds_combo_sweep.sh (afterok dependency).

set -eo pipefail
exec 2>&1

REPO="$PWD"
export PYTHONPATH="$REPO:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages"
PYTHON=/home/liyading/miniconda3/bin/python3

ROOT="/net/projects/CLS/lding/data/fa_data_analysis"
RUNS="$ROOT/ae_results/contrastive_run/ds_combo_enlcrop_sc2"
COMBO_LIST="config/contrastive_config/ds_combo/combo_list.txt"

mkdir -p logs/slurm

echo "======================================================================"
echo "Job   : $SLURM_JOB_ID  Node: $(hostname)"
echo "GPU   : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Start : $(date)"
echo "======================================================================"

# ── Stage 1: Cross-dataset eval (violin plots, all 4 ds) ─────────────────────
echo ""
echo "=== Stage 1: cross-dataset eval ==="
while IFS= read -r COMBO; do
    MODEL_DIR="$RUNS/$COMBO"
    if [ ! -f "$MODEL_DIR/model_final.pt" ] && [ ! -f "$MODEL_DIR/model_best.pt" ]; then
        echo "  SKIP $COMBO (no checkpoint)"
        continue
    fi
    echo ""
    echo "--- eval: $COMBO ---"
    $PYTHON scripts/run_cross_dataset_eval.py "$MODEL_DIR" \
        --mode sweep --root-folder "$ROOT"
done < "$COMBO_LIST"

# ── Stage 2: UMAP + cluster panels (z_proj) ──────────────────────────────────
echo ""
echo "=== Stage 2: UMAP + cluster analysis ==="
while IFS= read -r COMBO; do
    MODEL_DIR="$RUNS/$COMBO"
    LATENTS="$MODEL_DIR/eval/latents.csv"
    if [ ! -f "$LATENTS" ]; then
        echo "  SKIP $COMBO (no latents.csv)"
        continue
    fi
    echo ""
    echo "--- analysis: $COMBO ---"
    $PYTHON scripts/run_ds_combo_analysis.py "$MODEL_DIR"
done < "$COMBO_LIST"

echo ""
echo "End: $(date)"
echo "All done."
