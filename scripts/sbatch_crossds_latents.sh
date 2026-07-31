#!/usr/bin/env bash
#SBATCH --job-name=crossds_latents
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/crossds_latents_%j.out

# Compute cross-dataset latents + UMAP/PHATE embeddings for all ConAE models.
# Produces TWO embedding spaces per model:
#   z_recon (z_*): uses existing umap_model.pkl / umap_reducer.pkl / phate_model.pkl
#   z_proj  (p_*): fits fresh umap_proj_model.pkl / phate_proj_model.pkl
#
# Models covered:
#   contrastive/supcon cio_vinc lat12/lat18/lat24
#   ds_combo_enlcrop_clip01_l1  (15 combos)
#   ds_combo_enlcrop_sc2_clip02_l1  (15 combos)
#
# Note: cio_rb models are skipped (no checkpoint .pt available on cluster).

set -eo pipefail
exec 2>&1

REPO="$PWD"
PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
export PYTHONPATH="$REPO"
mkdir -p logs/slurm

RUNS="/net/projects/CLS/lding/data/fa_data_analysis/ae_results/contrastive_run"
COMBO_LIST="config/contrastive_config/ds_combo_v3/combo_list.txt"
CROSSDS="$PYTHON scripts/run_crossds_latents.py"

echo "======================================================================"
echo "Job:   $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"
echo "======================================================================"

# ── cio_vinc lat12 / lat18 / lat24 ───────────────────────────────────────────
echo ""
echo "=== cio_vinc flat models ==="
for MODEL_DIR in \
    "$RUNS"/contrastive_cio_vinc_lat12proj8_enlcrop_sc2* \
    "$RUNS"/supcon_cio_vinc_lat12proj8_enlcrop_sc2* \
    "$RUNS"/contrastive_cio_vinc_lat18proj12_enlcrop_sc2* \
    "$RUNS"/contrastive_cio_vinc_lat24proj16_enlcrop_sc2*
do
    [ -d "$MODEL_DIR" ] || continue
    [ -f "$MODEL_DIR/latents.csv" ] || continue
    echo ""
    echo "--- $(basename $MODEL_DIR) ---"
    $CROSSDS "$MODEL_DIR" || echo "  WARNING: failed for $(basename $MODEL_DIR)"
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
        $CROSSDS "$MODEL_DIR" || echo "  WARNING: failed for $COMBO"
    done < "$COMBO_LIST"
done

echo ""
echo "End: $(date)"
