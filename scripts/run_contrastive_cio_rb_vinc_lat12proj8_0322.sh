#!/usr/bin/env bash
# =============================================================================
# Full pipeline: contrastive AE on vinc CIO-RB, latent=12 proj=8
#   Training strategy: 0322 (200 epochs, no weight decay, no LR scheduler)
#   Stage 1: AE training
#   Stage 2: Analysis (UMAP on z_recon and z_proj)
#   Stage 3: Classification (fa + pos, z_recon and z_proj)
#   Stage 4: Cross-classification visualization (z_recon, z_proj)
#
# Results: ae_results/contrastive_run/contrastive_cio_rb_vinc_lat12proj8_0322/
# =============================================================================

set -euo pipefail

export PYTHONPATH="/net/projects/CLS/lding/gitcode/SubCellAE_contrastive_projector"
PYTHON="/home/liyading/miniconda3/envs/subcellae-cuda/bin/python"
CFG="config/contrastive_config"

mkdir -p logs

echo "======================================================================"
echo " STAGE 1 — AE training (vinc CIO-RB, latent=12, proj=8, 0322 strategy)"
echo "======================================================================"
$PYTHON scripts/run_ae_from_config.py \
    $CFG/ae_contrastive_cio_rb_vinc_lat12proj8_0322.yaml

echo ""
echo "======================================================================"
echo " STAGE 2 — Analysis (UMAP z_recon + z_proj)"
echo "======================================================================"
$PYTHON scripts/run_analysis_from_config.py \
    $CFG/analysis_contrastive_cio_rb_vinc_lat12proj8_0322.yaml

echo ""
echo "======================================================================"
echo " STAGE 3 — Classification  (4 runs: 2 targets × 2 feature sets)"
echo "======================================================================"

echo "--- [1/4] FA type  | z_recon ---"
$PYTHON scripts/run_classification_from_config.py \
    $CFG/cls_contrastive_cio_rb_vinc_lat12proj8_0322_fa_zrecon.yaml

echo "--- [2/4] FA type  | z_proj  ---"
$PYTHON scripts/run_classification_from_config.py \
    $CFG/cls_contrastive_cio_rb_vinc_lat12proj8_0322_fa_zproj.yaml

echo "--- [3/4] Position | z_recon ---"
$PYTHON scripts/run_classification_from_config.py \
    $CFG/cls_contrastive_cio_rb_vinc_lat12proj8_0322_pos_zrecon.yaml

echo "--- [4/4] Position | z_proj  ---"
$PYTHON scripts/run_classification_from_config.py \
    $CFG/cls_contrastive_cio_rb_vinc_lat12proj8_0322_pos_zproj.yaml

echo ""
echo "======================================================================"
echo " STAGE 4 — Visualization  (2 runs: z_recon, z_proj)"
echo "======================================================================"

echo "--- [1/2] z_recon ---"
$PYTHON scripts/run_cross_classification_vis.py \
    $CFG/vis_contrastive_cio_rb_vinc_lat12proj8_0322_zrecon.yaml

echo "--- [2/2] z_proj  ---"
$PYTHON scripts/run_cross_classification_vis.py \
    $CFG/vis_contrastive_cio_rb_vinc_lat12proj8_0322_zproj.yaml

echo ""
echo "======================================================================"
echo " ALL DONE — results in ae_results/contrastive_run/contrastive_cio_rb_vinc_lat12proj8_0322/"
echo "======================================================================"
