#!/usr/bin/env bash
#SBATCH --job-name=quality_panels_4ds
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/quality_panels_4ds_%j.out

# Reconstruction quality panels using GLOBAL percentile thresholds from all
# 4 datasets combined (vinc + pfak + ppax + nih3t3).
# Covers all enlcrop_sc2 con + supcon models.
# Output: <model_dir>/quality_panels_4ds_bulk/

set -eo pipefail
exec 2>&1

REPO="$PWD"
PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
export PYTHONPATH="$REPO"
mkdir -p logs/slurm

ROOT="/net/projects/CLS/lding/data/fa_data_analysis"
RUNS="$ROOT/ae_results/contrastive_run"

echo "======================================================================"
echo "Job:   $SLURM_JOB_ID  Node: $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Start: $(date)"
echo "======================================================================"

run_panels() {
    local MODEL_DIR="$1"
    echo ""
    echo "--- $(basename $MODEL_DIR) ---"
    $PYTHON scripts/make_recon_quality_panels_4ds.py "$MODEL_DIR" \
        --root-folder "$ROOT"
}

# --- contrastive (ConAE) enlcrop_sc2 vinc-only ---
run_panels "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2"
run_panels "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_hessian_lc025"
run_panels "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_l1"
run_panels "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_l1_lc025"
run_panels "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_lr4"
run_panels "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_lr8"
run_panels "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_mse_lc025"
run_panels "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1"
run_panels "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_lc0031"
run_panels "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_lc0062"
run_panels "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_lc0125"
run_panels "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_lc025"
run_panels "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nmse"
run_panels "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nmse_lc0125"
run_panels "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nmse_lc025"

# --- contrastive (ConAE) enlcrop_sc2 vinc+ppax ---
run_panels "$RUNS/contrastive_cio_rb_vinc_ppax_lat12proj8_enlcrop_sc2_nl1"
run_panels "$RUNS/contrastive_cio_rb_vinc_ppax_lat12proj8_enlcrop_sc2_nl1_lc025"

# --- supervised contrastive (SupConAE) enlcrop_sc2 vinc-only ---
run_panels "$RUNS/supcon_cio_rb_vinc_lat12proj8_enlcrop_sc2"
run_panels "$RUNS/supcon_cio_rb_vinc_lat12proj8_enlcrop_sc2_l1"
run_panels "$RUNS/supcon_cio_rb_vinc_lat12proj8_enlcrop_sc2_lr4"
run_panels "$RUNS/supcon_cio_rb_vinc_lat12proj8_enlcrop_sc2_lr8"
run_panels "$RUNS/supcon_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1"
run_panels "$RUNS/supcon_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_lc0125"
run_panels "$RUNS/supcon_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_lc025"
run_panels "$RUNS/supcon_cio_rb_vinc_lat12proj8_enlcrop_sc2_nmse"
run_panels "$RUNS/supcon_cio_rb_vinc_lat12proj8_enlcrop_sc2_nmse_lc0125"
run_panels "$RUNS/supcon_cio_rb_vinc_lat12proj8_enlcrop_sc2_nmse_lc025"

# --- supervised contrastive (SupConAE) enlcrop_sc2 vinc+ppax ---
run_panels "$RUNS/supcon_cio_rb_vinc_ppax_lat12proj8_enlcrop_sc2_l1"
run_panels "$RUNS/supcon_cio_rb_vinc_ppax_lat12proj8_enlcrop_sc2_nl1_lc025"

echo ""
echo "End: $(date)"
