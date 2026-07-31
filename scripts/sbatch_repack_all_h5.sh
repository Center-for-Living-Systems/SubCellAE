#!/usr/bin/env bash
#SBATCH --job-name=repack_all_h5
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/repack_all_h5_%j.out

# Re-pack ALL model.h5 files after run_crossds_latents has been run.
# Adds: eval/cross_dataset_latents.csv, 4ds UMAP/PHATE plots (both z_recon and
#       z_proj), and all top-level PNGs (loss curves, cross_dataset_recon_*.png).
#
# Covers ALL flat model dirs in contrastive_run that have latents.csv,
# plus all ds_combo_v3 per-combo subdirs.

set -eo pipefail
exec 2>&1

REPO="$PWD"
PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
export PYTHONPATH="$REPO"
mkdir -p logs/slurm

RUNS="/net/projects/CLS/lding/data/fa_data_analysis/ae_results/contrastive_run"
COMBO_LIST="config/contrastive_config/ds_combo_v3/combo_list.txt"
PACK="$PYTHON scripts/pack_conae_run_h5.py"

echo "======================================================================"
echo "Job:   $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"
echo "======================================================================"

# ── all flat model dirs (anything directly under contrastive_run with latents.csv)
echo ""
echo "=== flat contrastive_run/* model dirs ==="
for MODEL_DIR in "$RUNS"/*/; do
    [ -f "$MODEL_DIR/latents.csv" ] || continue
    echo ""
    echo "--- $(basename $MODEL_DIR) ---"
    $PACK "$MODEL_DIR" || echo "  WARNING: pack failed for $(basename $MODEL_DIR)"
done

# ── all ds_combo_* per-combo subdirs ─────────────────────────────────────────
for PARENT_DIR in "$RUNS"/ds_combo_*/; do
    PARENT_NAME=$(basename "$PARENT_DIR")
    echo ""
    echo "=== $PARENT_NAME ==="
    while IFS= read -r COMBO; do
        MODEL_DIR="$PARENT_DIR/$COMBO"
        [ -f "$MODEL_DIR/latents.csv" ] || { echo "  SKIP $COMBO (no latents.csv)"; continue; }
        echo ""
        echo "--- $PARENT_NAME / $COMBO ---"
        $PACK "$MODEL_DIR" || echo "  WARNING: pack failed for $COMBO"
    done < "$COMBO_LIST"
done

echo ""
echo "End: $(date)"
