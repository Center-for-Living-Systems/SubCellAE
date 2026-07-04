#!/usr/bin/env bash
#SBATCH --job-name=cluster_panels_enlcrop
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/cluster_panels_enlcrop_%j.out

# KMeans k=10 + 16-patch center panels + UMAP/PHATE scatter plots
# for all enlcrop_sc2 contrastive and supervised-contrastive models (excluding ch3).

set -eo pipefail
exec 2>&1

REPO="$PWD"
PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
export PYTHONPATH="$REPO"
mkdir -p logs/slurm

RUNS="/net/projects/CLS/lding/data/fa_data_analysis/ae_results/contrastive_run"

echo "======================================================================"
echo "Job:   $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"
echo "======================================================================"

# --- contrastive (ConAE) enlcrop_sc2 vinc-only ---
for MODEL_DIR in \
    "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2" \
    "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_hessian_lc025" \
    "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_l1" \
    "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_l1_lc025" \
    "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_lr4" \
    "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_lr8" \
    "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_mse_lc025" \
    "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1" \
    "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_lc0031" \
    "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_lc0062" \
    "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_lc0125" \
    "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_lc025" \
    "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nmse" \
    "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nmse_lc0125" \
    "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nmse_lc025"
do
    echo ""
    echo "--- $(basename $MODEL_DIR) ---"
    $PYTHON scripts/run_cluster_panels.py "$MODEL_DIR" --k 10 --n-panel 16
done

# --- contrastive (ConAE) enlcrop_sc2 vinc+ppax ---
for MODEL_DIR in \
    "$RUNS/contrastive_cio_rb_vinc_ppax_lat12proj8_enlcrop_sc2_nl1" \
    "$RUNS/contrastive_cio_rb_vinc_ppax_lat12proj8_enlcrop_sc2_nl1_lc025"
do
    echo ""
    echo "--- $(basename $MODEL_DIR) ---"
    $PYTHON scripts/run_cluster_panels.py "$MODEL_DIR" --k 10 --n-panel 16
done

# --- supervised contrastive (SupConAE) enlcrop_sc2 vinc-only ---
for MODEL_DIR in \
    "$RUNS/supcon_cio_rb_vinc_lat12proj8_enlcrop_sc2" \
    "$RUNS/supcon_cio_rb_vinc_lat12proj8_enlcrop_sc2_l1" \
    "$RUNS/supcon_cio_rb_vinc_lat12proj8_enlcrop_sc2_lr4" \
    "$RUNS/supcon_cio_rb_vinc_lat12proj8_enlcrop_sc2_lr8" \
    "$RUNS/supcon_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1" \
    "$RUNS/supcon_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_lc0125" \
    "$RUNS/supcon_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_lc025" \
    "$RUNS/supcon_cio_rb_vinc_lat12proj8_enlcrop_sc2_nmse" \
    "$RUNS/supcon_cio_rb_vinc_lat12proj8_enlcrop_sc2_nmse_lc0125" \
    "$RUNS/supcon_cio_rb_vinc_lat12proj8_enlcrop_sc2_nmse_lc025"
do
    echo ""
    echo "--- $(basename $MODEL_DIR) ---"
    $PYTHON scripts/run_cluster_panels.py "$MODEL_DIR" --k 10 --n-panel 16
done

# --- supervised contrastive (SupConAE) enlcrop_sc2 vinc+ppax ---
for MODEL_DIR in \
    "$RUNS/supcon_cio_rb_vinc_ppax_lat12proj8_enlcrop_sc2_l1" \
    "$RUNS/supcon_cio_rb_vinc_ppax_lat12proj8_enlcrop_sc2_nl1_lc025"
do
    echo ""
    echo "--- $(basename $MODEL_DIR) ---"
    $PYTHON scripts/run_cluster_panels.py "$MODEL_DIR" --k 10 --n-panel 16
done

echo ""
echo "End: $(date)"
