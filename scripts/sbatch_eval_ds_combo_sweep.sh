#!/usr/bin/env bash
#SBATCH --job-name=ds_combo_eval
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/ds_combo_eval_%j.out

# Post-training eval + analysis for all 15 dataset-combo models.
#   Stage 1: run_cross_dataset_eval.py on the whole ds_combo_enlcrop_sc2/ dir
#            (sweep mode discovers all 15 models, saves per-model violin plots
#             + combined cross_dataset_recon_metrics.csv)
#   Stage 2: UMAP + KMeans cluster panels per model (run_ds_combo_analysis.py)

set -o pipefail
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

# ── Stage 1: Cross-dataset eval (all 15 models in one sweep) ─────────────────
echo ""
echo "=== Stage 1: cross-dataset eval (sweep) ==="
$PYTHON scripts/run_cross_dataset_eval.py "$RUNS" \
    --mode sweep --root-folder "$ROOT"

# ── Stage 2: UMAP + cluster panels per model ─────────────────────────────────
echo ""
echo "=== Stage 2: UMAP + cluster analysis ==="
while IFS= read -r COMBO; do
    MODEL_DIR="$RUNS/$COMBO"
    LATENTS="$MODEL_DIR/latents.csv"
    if [ ! -f "$LATENTS" ]; then
        echo "  SKIP $COMBO (no latents.csv)"
        continue
    fi
    echo ""
    echo "--- analysis: $COMBO ---"
    $PYTHON scripts/run_ds_combo_analysis.py "$MODEL_DIR" || echo "  WARNING: analysis failed for $COMBO"
done < "$COMBO_LIST"

echo ""
echo "End: $(date)"
echo "All done."
