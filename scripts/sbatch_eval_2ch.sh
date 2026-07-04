#!/usr/bin/env bash
#SBATCH --job-name=eval_2ch
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/eval_2ch_%j.out
# patchprep 944484 completed; no dependency needed

# Comprehensive evaluation for 2-channel (pax+actin) and actin-only AE models.
# Depends on patchprep job 944429 (ch3 patches for ppax/pfak/nih3t3).
#
# Steps:
#  A. 2ch models: UMAP + KNN cls + ppax cross-dataset (--two-channel)
#  B. 2ch models: cross-dataset recon metrics violin plots (all 3 external datasets)
#  C. 2ch models: 10-90% recon quality panels (vinc unlabelled + external)
#  D. actin-only models: UMAP + cross-dataset recon metrics
#  E. actin-only models: 10-90% recon quality panels (vinc unlabelled + external)

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

# ── 2ch model dirs ────────────────────────────────────────────────────────────
BASELINE_2CH="$RUNS/baseline_vinc_2ch_pax_act"
CONAE_NL1_2CH="$RUNS/contrastive_cio_rb_vinc_lat12proj8_sc2_nl1_2ch_pax_act"
CONAE_NL1_LC025_2CH="$RUNS/contrastive_cio_rb_vinc_lat12proj8_sc2_nl1_lc025_2ch_pax_act"

# ── actin-only model dirs ─────────────────────────────────────────────────────
BASELINE_ACT="$RUNS/baseline_vinc_only_ch3"
CONAE_NL1_ACT="$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_ch3"
CONAE_NL1_LC025_ACT="$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_lc025_ch3"

# ── patch dirs ────────────────────────────────────────────────────────────────
PPAX_CH1_CTRL="$PATCHES/ppax/control/tiff_patches32_mr10"
PPAX_CH1_YCOMP="$PATCHES/ppax/ycomp/tiff_patches32_mr10"
PPAX_CH3_CTRL="$PATCHES/ppax_ch3/control/tiff_patches32_mr10"
PPAX_CH3_YCOMP="$PATCHES/ppax_ch3/ycomp/tiff_patches32_mr10"

PFAK_CH1_CTRL="$PATCHES/pfak/control/tiff_patches32_mr10"
PFAK_CH1_YCOMP="$PATCHES/pfak/ycomp/tiff_patches32_mr10"
PFAK_CH3_CTRL="$PATCHES/pfak_ch3/control/tiff_patches32_mr10"
PFAK_CH3_YCOMP="$PATCHES/pfak_ch3/ycomp/tiff_patches32_mr10"

NIH_CH1_CTRL="$PATCHES/nih3t3/control/tiff_patches32_mr10"
NIH_CH1_YCOMP="$PATCHES/nih3t3/ycomp/tiff_patches32_mr10"
NIH_CH3_CTRL="$PATCHES/nih3t3_ch3/control/tiff_patches32_mr10"
NIH_CH3_YCOMP="$PATCHES/nih3t3_ch3/ycomp/tiff_patches32_mr10"

VINC_CH3_CTRL="$PATCHES/vinc_ch3/control/tiff_patches32_mr10"
VINC_CH3_YCOMP="$PATCHES/vinc_ch3/ycomp/tiff_patches32_mr10"

# =============================================================================
# A. 2ch models — UMAP + KNN classification + ppax cross-dataset
# =============================================================================
echo ""
echo "### A. 2ch: UMAP + KNN + ppax cross-dataset ###"

for MODEL_DIR in "$BASELINE_2CH" "$CONAE_NL1_2CH" "$CONAE_NL1_LC025_2CH"; do
    echo ""
    echo "--- run_contrastive_eval (2ch): $(basename $MODEL_DIR) ---"
    $PYTHON scripts/run_contrastive_eval.py "$MODEL_DIR" \
        --batch_size 512 --two-channel
done

# =============================================================================
# B. 2ch models — cross-dataset recon quality violin plots (all external ds)
# =============================================================================
echo ""
echo "### B. 2ch: cross-dataset recon metrics (pfak/ppax/nih3t3) ###"

for MODEL_DIR in "$BASELINE_2CH" "$CONAE_NL1_2CH" "$CONAE_NL1_LC025_2CH"; do
    echo ""
    echo "--- run_cross_dataset_eval (2ch): $(basename $MODEL_DIR) ---"
    $PYTHON scripts/run_cross_dataset_eval.py "$MODEL_DIR" \
        --mode sweep --two-channel --root-folder "$ROOT"
done

# =============================================================================
# C. 2ch models — 10-90% recon quality panels
# =============================================================================
echo ""
echo "### C. 2ch: recon quality panels (vinc unlabelled + external) ###"

for MODEL_DIR in "$BASELINE_2CH" "$CONAE_NL1_2CH" "$CONAE_NL1_LC025_2CH"; do
    echo ""
    echo "--- panels (vinc unlabelled): $(basename $MODEL_DIR) ---"
    $PYTHON scripts/make_recon_quality_panels_bulk.py "$MODEL_DIR" \
        --subset unlabelled

    echo "--- panels (ppax 2ch): $(basename $MODEL_DIR) ---"
    $PYTHON scripts/make_recon_quality_panels_bulk.py "$MODEL_DIR" \
        --subset ppax \
        --patch-dirs    "$PPAX_CH1_CTRL" "$PPAX_CH1_YCOMP" \
        --ch3-patch-dirs "$PPAX_CH3_CTRL" "$PPAX_CH3_YCOMP"

    echo "--- panels (pfak 2ch): $(basename $MODEL_DIR) ---"
    $PYTHON scripts/make_recon_quality_panels_bulk.py "$MODEL_DIR" \
        --subset pfak \
        --patch-dirs    "$PFAK_CH1_CTRL" "$PFAK_CH1_YCOMP" \
        --ch3-patch-dirs "$PFAK_CH3_CTRL" "$PFAK_CH3_YCOMP"

    echo "--- panels (nih3t3 2ch): $(basename $MODEL_DIR) ---"
    $PYTHON scripts/make_recon_quality_panels_bulk.py "$MODEL_DIR" \
        --subset nih3t3 \
        --patch-dirs    "$NIH_CH1_CTRL" "$NIH_CH1_YCOMP" \
        --ch3-patch-dirs "$NIH_CH3_CTRL" "$NIH_CH3_YCOMP"
done

# =============================================================================
# D. actin-only models — UMAP + cross-dataset recon metrics (single ch3)
# =============================================================================
echo ""
echo "### D. actin-only: UMAP + cross-dataset recon metrics ###"

for MODEL_DIR in "$BASELINE_ACT" "$CONAE_NL1_ACT" "$CONAE_NL1_LC025_ACT"; do
    echo ""
    echo "--- run_contrastive_eval (actin-only): $(basename $MODEL_DIR) ---"
    # No --two-channel: runs UMAP/cluster but skips KNN (no pax labels)
    $PYTHON scripts/run_contrastive_eval.py "$MODEL_DIR" --batch_size 512

    echo "--- cross-dataset recon metrics (ch3 external): $(basename $MODEL_DIR) ---"
    # Use single-channel external patches in the cio_rb path
    $PYTHON scripts/run_cross_dataset_eval.py "$MODEL_DIR" \
        --mode sweep --root-folder "$ROOT"
done

# =============================================================================
# E. actin-only models — 10-90% recon quality panels
# =============================================================================
echo ""
echo "### E. actin-only: recon quality panels ###"

for MODEL_DIR in "$BASELINE_ACT" "$CONAE_NL1_ACT" "$CONAE_NL1_LC025_ACT"; do
    echo ""
    echo "--- panels (vinc unlabelled actin): $(basename $MODEL_DIR) ---"
    $PYTHON scripts/make_recon_quality_panels_bulk.py "$MODEL_DIR" \
        --subset unlabelled

    echo "--- panels (ppax actin ch3): $(basename $MODEL_DIR) ---"
    $PYTHON scripts/make_recon_quality_panels_bulk.py "$MODEL_DIR" \
        --subset ppax \
        --patch-dirs "$PPAX_CH3_CTRL" "$PPAX_CH3_YCOMP"

    echo "--- panels (pfak actin ch3): $(basename $MODEL_DIR) ---"
    $PYTHON scripts/make_recon_quality_panels_bulk.py "$MODEL_DIR" \
        --subset pfak \
        --patch-dirs "$PFAK_CH3_CTRL" "$PFAK_CH3_YCOMP"

    echo "--- panels (nih3t3 actin ch3): $(basename $MODEL_DIR) ---"
    $PYTHON scripts/make_recon_quality_panels_bulk.py "$MODEL_DIR" \
        --subset nih3t3 \
        --patch-dirs "$NIH_CH3_CTRL" "$NIH_CH3_YCOMP"
done

echo ""
echo "End: $(date)"
