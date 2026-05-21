#!/usr/bin/env bash
# =============================================================================
# Stages 3–4 only: Classification + Visualization
#   Uses existing latents from a completed Stage 1+2 run.
#   Re-runs classification with updated label CSV and regenerates vis.
#
# Results: ae_results/contrastive_run/contrastive_cio_rb_vinc_lat12proj8/
#
# Usage:
#   conda activate subcellae-cuda
#   cd /path/to/SubCellAE-contrastive
#   bash scripts/run_contrastive_cio_rb_vinc_lat12proj8_cls_vis.sh
# =============================================================================

set -euo pipefail

export PYTHONPATH="/net/projects/CLS/lding/gitcode/SubCellAE_contrastive_projector"
PYTHON="/home/liyading/miniconda3/envs/subcellae-cuda/bin/python"
CFG="config/contrastive_config"

mkdir -p logs

echo "======================================================================"
echo " STAGE 3 — Classification  (4 runs: 2 targets × 2 feature sets)"
echo "======================================================================"

echo "--- [1/4] FA type  | z_recon ---"
$PYTHON scripts/run_classification_from_config.py \
    $CFG/cls_contrastive_cio_rb_vinc_lat12proj8_fa_zrecon.yaml

echo "--- [2/4] FA type  | z_proj  ---"
$PYTHON scripts/run_classification_from_config.py \
    $CFG/cls_contrastive_cio_rb_vinc_lat12proj8_fa_zproj.yaml

echo "--- [3/4] Position | z_recon ---"
$PYTHON scripts/run_classification_from_config.py \
    $CFG/cls_contrastive_cio_rb_vinc_lat12proj8_pos_zrecon.yaml

echo "--- [4/4] Position | z_proj  ---"
$PYTHON scripts/run_classification_from_config.py \
    $CFG/cls_contrastive_cio_rb_vinc_lat12proj8_pos_zproj.yaml

echo ""
echo "======================================================================"
echo " STAGE 4 — Visualization  (2 runs: z_recon, z_proj)"
echo "======================================================================"

echo "--- [1/2] z_recon ---"
$PYTHON scripts/run_cross_classification_vis.py \
    $CFG/vis_contrastive_cio_rb_vinc_lat12proj8_zrecon.yaml

echo "--- [2/2] z_proj  ---"
$PYTHON scripts/run_cross_classification_vis.py \
    $CFG/vis_contrastive_cio_rb_vinc_lat12proj8_zproj.yaml

echo ""
echo "======================================================================"
echo " STAGES 3–4 DONE"
echo "======================================================================"
