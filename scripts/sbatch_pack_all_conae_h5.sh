#!/usr/bin/env bash
#SBATCH --job-name=pack_conae_h5
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/pack_all_conae_h5_%j.out

# Pack every ConAE / SupConAE model result into model.h5.
# Covers: all contrastive_cio_rb_*, supcon_cio_rb_*, contrastive_cio_vinc_*,
#         supcon_cio_vinc_*, and ds_combo_v3 per-combo subdirs.
# Run AFTER cluster panel jobs (1216881, 1216882) finish so panels are included.

set -eo pipefail
exec 2>&1

REPO="$PWD"
PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
export PYTHONPATH="$REPO"
mkdir -p logs/slurm

RUNS="/net/projects/CLS/lding/data/fa_data_analysis/ae_results/contrastive_run"
PACK="$PYTHON scripts/pack_conae_run_h5.py"

echo "======================================================================"
echo "Job:   $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"
echo "======================================================================"

# ── flat model directories (cio_rb + cio_vinc) ────────────────────────────────
echo ""
echo "=== flat contrastive_run/* model dirs ==="
for MODEL_DIR in \
    "$RUNS"/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2* \
    "$RUNS"/supcon_cio_rb_vinc_lat12proj8_enlcrop_sc2* \
    "$RUNS"/contrastive_cio_rb_vinc_ppax_lat12proj8_enlcrop_sc2* \
    "$RUNS"/supcon_cio_rb_vinc_ppax_lat12proj8_enlcrop_sc2* \
    "$RUNS"/contrastive_cio_rb_pfak_lat12proj8_enlcrop_sc2* \
    "$RUNS"/contrastive_cio_rb_ppax_lat12proj8_enlcrop_sc2* \
    "$RUNS"/contrastive_cio_vinc_lat12proj8_enlcrop_sc2* \
    "$RUNS"/supcon_cio_vinc_lat12proj8_enlcrop_sc2* \
    "$RUNS"/contrastive_cio_vinc_lat18proj12_enlcrop_sc2* \
    "$RUNS"/contrastive_cio_vinc_lat24proj16_enlcrop_sc2*
do
    [ -d "$MODEL_DIR" ] || continue
    [ -f "$MODEL_DIR/latents.csv" ] || continue
    echo ""
    echo "--- $(basename $MODEL_DIR) ---"
    $PACK "$MODEL_DIR" || echo "  WARNING: pack failed for $(basename $MODEL_DIR)"
done

# ── ds_combo_v3 per-combo subdirs ─────────────────────────────────────────────
COMBO_LIST="config/contrastive_config/ds_combo_v3/combo_list.txt"

echo ""
echo "=== ds_combo_enlcrop_clip01_l1 ==="
while IFS= read -r COMBO; do
    MODEL_DIR="$RUNS/ds_combo_enlcrop_clip01_l1/$COMBO"
    [ -f "$MODEL_DIR/latents.csv" ] || { echo "  SKIP $COMBO (no latents.csv)"; continue; }
    echo ""
    echo "--- clip01_l1 / $COMBO ---"
    $PACK "$MODEL_DIR" || echo "  WARNING: pack failed for $COMBO"
done < "$COMBO_LIST"

echo ""
echo "=== ds_combo_enlcrop_sc2_clip02_l1 ==="
while IFS= read -r COMBO; do
    MODEL_DIR="$RUNS/ds_combo_enlcrop_sc2_clip02_l1/$COMBO"
    [ -f "$MODEL_DIR/latents.csv" ] || { echo "  SKIP $COMBO (no latents.csv)"; continue; }
    echo ""
    echo "--- sc2_clip02_l1 / $COMBO ---"
    $PACK "$MODEL_DIR" || echo "  WARNING: pack failed for $COMBO"
done < "$COMBO_LIST"

echo ""
echo "End: $(date)"
