#!/usr/bin/env bash
#SBATCH --job-name=eval_vinc_ppax
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/eval_vinc_ppax_%j.out

# Cross-dataset recon violin plots + 10-90% quality panels for the 4 models
# trained jointly on vinc + ppax (ds1 + ds3).
#
# UMAP / PHATE / confusion matrices already exist in eval/ for all 4 models.
# Steps:
#  A. Cross-dataset recon metric violin plots (pfak, nih3t3; ppax as in-dist)
#  B. 10-90% recon quality panels (vinc unlabelled + ppax/pfak/nih3t3)

set -eo pipefail
exec 2>&1

REPO="$PWD"
PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
export PYTHONPATH="$REPO"
mkdir -p logs/slurm

ROOT="/net/projects/CLS/lding/data/fa_data_analysis"
RUNS="$ROOT/ae_results/contrastive_run"
PATCHES="$ROOT/ae_results/patches/cio_rb"

echo "======================================================================"
echo "Job:   $SLURM_JOB_ID"
echo "Node:  $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Start: $(date)"
echo "======================================================================"

# ── model dirs ────────────────────────────────────────────────────────────────
CON_NL1="$RUNS/contrastive_cio_rb_vinc_ppax_lat12proj8_enlcrop_sc2_nl1"
CON_NL1_LC025="$RUNS/contrastive_cio_rb_vinc_ppax_lat12proj8_enlcrop_sc2_nl1_lc025"
SC_L1="$RUNS/supcon_cio_rb_vinc_ppax_lat12proj8_enlcrop_sc2_l1"
SC_NL1_LC025="$RUNS/supcon_cio_rb_vinc_ppax_lat12proj8_enlcrop_sc2_nl1_lc025"

# ── patch dirs (ch1 = paxillin) ───────────────────────────────────────────────
PPAX_CTRL="$PATCHES/ppax/control/tiff_patches32_mr10"
PPAX_YCOMP="$PATCHES/ppax/ycomp/tiff_patches32_mr10"

PFAK_CTRL="$PATCHES/pfak/control/tiff_patches32_mr10"
PFAK_YCOMP="$PATCHES/pfak/ycomp/tiff_patches32_mr10"

NIH_CTRL="$PATCHES/nih3t3/control/tiff_patches32_mr10"
NIH_YCOMP="$PATCHES/nih3t3/ycomp/tiff_patches32_mr10"

# =============================================================================
# A. Cross-dataset recon metric violin plots
# =============================================================================
echo ""
echo "### A. Cross-dataset recon metrics (pfak / ppax / nih3t3) ###"

for MODEL_DIR in "$CON_NL1" "$CON_NL1_LC025" "$SC_L1" "$SC_NL1_LC025"; do
    echo ""
    echo "--- run_cross_dataset_eval: $(basename $MODEL_DIR) ---"
    $PYTHON scripts/run_cross_dataset_eval.py "$MODEL_DIR" \
        --mode sweep --root-folder "$ROOT"
done

# =============================================================================
# B. 10-90% recon quality panels
# =============================================================================
echo ""
echo "### B. Recon quality panels (vinc unlabelled + ppax / pfak / nih3t3) ###"

for MODEL_DIR in "$CON_NL1" "$CON_NL1_LC025" "$SC_L1" "$SC_NL1_LC025"; do
    echo ""
    echo "--- panels (vinc unlabelled): $(basename $MODEL_DIR) ---"
    $PYTHON scripts/make_recon_quality_panels_bulk.py "$MODEL_DIR" \
        --subset unlabelled

    echo "--- panels (ppax): $(basename $MODEL_DIR) ---"
    $PYTHON scripts/make_recon_quality_panels_bulk.py "$MODEL_DIR" \
        --subset ppax \
        --patch-dirs "$PPAX_CTRL" "$PPAX_YCOMP"

    echo "--- panels (pfak): $(basename $MODEL_DIR) ---"
    $PYTHON scripts/make_recon_quality_panels_bulk.py "$MODEL_DIR" \
        --subset pfak \
        --patch-dirs "$PFAK_CTRL" "$PFAK_YCOMP"

    echo "--- panels (nih3t3): $(basename $MODEL_DIR) ---"
    $PYTHON scripts/make_recon_quality_panels_bulk.py "$MODEL_DIR" \
        --subset nih3t3 \
        --patch-dirs "$NIH_CTRL" "$NIH_YCOMP"
done

echo ""
echo "End: $(date)"
