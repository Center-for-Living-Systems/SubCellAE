#!/usr/bin/env bash
#SBATCH --job-name=eval_loss_lc
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/eval_loss_lc_%j.out

# Eval pipeline for 5 new ConAE models (loss sweep + lambda sweep on vinc).
# Steps:
#  A. UMAP + KNN classification + ppax cross-dataset
#  B. Cross-dataset recon metric violin plots
#  C. 10-90% recon quality panels (vinc unlabelled + ppax/pfak/nih3t3)

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
MSE_LC025="$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_mse_lc025"
L1_LC025="$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_l1_lc025"
HESSIAN_LC025="$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_hessian_lc025"
NL1_LC0062="$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_lc0062"
NL1_LC0031="$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_lc0031"

# ── patch dirs ────────────────────────────────────────────────────────────────
PPAX_CTRL="$PATCHES/ppax/control/tiff_patches32_mr10"
PPAX_YCOMP="$PATCHES/ppax/ycomp/tiff_patches32_mr10"
PFAK_CTRL="$PATCHES/pfak/control/tiff_patches32_mr10"
PFAK_YCOMP="$PATCHES/pfak/ycomp/tiff_patches32_mr10"
NIH_CTRL="$PATCHES/nih3t3/control/tiff_patches32_mr10"
NIH_YCOMP="$PATCHES/nih3t3/ycomp/tiff_patches32_mr10"

# =============================================================================
# A. UMAP + KNN classification + ppax cross-dataset
# =============================================================================
echo ""
echo "### A. UMAP + KNN + ppax cross-dataset ###"

for MODEL_DIR in "$MSE_LC025" "$L1_LC025" "$HESSIAN_LC025" "$NL1_LC0062" "$NL1_LC0031"; do
    echo ""
    echo "--- run_contrastive_eval: $(basename $MODEL_DIR) ---"
    $PYTHON scripts/run_contrastive_eval.py "$MODEL_DIR" --batch_size 512
done

# =============================================================================
# B. Cross-dataset recon metric violin plots
# =============================================================================
echo ""
echo "### B. Cross-dataset recon metrics (ppax / pfak / nih3t3) ###"

for MODEL_DIR in "$MSE_LC025" "$L1_LC025" "$HESSIAN_LC025" "$NL1_LC0062" "$NL1_LC0031"; do
    echo ""
    echo "--- run_cross_dataset_eval: $(basename $MODEL_DIR) ---"
    $PYTHON scripts/run_cross_dataset_eval.py "$MODEL_DIR" \
        --mode sweep --root-folder "$ROOT"
done

# =============================================================================
# C. 10-90% recon quality panels
# =============================================================================
echo ""
echo "### C. Recon quality panels (vinc unlabelled + ppax / pfak / nih3t3) ###"

for MODEL_DIR in "$MSE_LC025" "$L1_LC025" "$HESSIAN_LC025" "$NL1_LC0062" "$NL1_LC0031"; do
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
