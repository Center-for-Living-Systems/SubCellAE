#!/usr/bin/env bash
#SBATCH --job-name=panels_cio_missing
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=logs/slurm/cluster_panels_cio_missing_%j.out

# KMeans k=10 cluster panels for cio_rb and cio_vinc models missing panels.
# Covers: 12 contrastive_cio_rb shift/4ch variants + 10 supcon_cio_rb shift variants
#       + 3 contrastive_cio_vinc lat12 + 3 supcon_cio_vinc lat12  (28 models total)

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

# --- contrastive_cio_rb missing (shift / 4ch variants) ---
for MODEL_DIR in \
    "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_mse_shift0" \
    "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_mse_shift0_rot0" \
    "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_mse_shift0_rot0_nosig" \
    "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_mse_shift0_rot0_nosig_warmup100" \
    "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_mse_shift0_rot0_warmup100" \
    "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_mse_shift1_nojitter" \
    "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_4ch_vinc" \
    "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_lc025_4ch_vinc" \
    "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_shift0" \
    "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_shift0_rot0" \
    "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_shift0_rot0_lc003" \
    "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_shift1_nojitter"
do
    echo ""
    echo "--- $(basename $MODEL_DIR) ---"
    $PYTHON scripts/run_cluster_panels.py "$MODEL_DIR" --k 10 --n-panel 16
done

# --- supcon_cio_rb missing (shift variants) ---
for MODEL_DIR in \
    "$RUNS/supcon_cio_rb_vinc_lat12proj8_enlcrop_sc2_mse_shift0" \
    "$RUNS/supcon_cio_rb_vinc_lat12proj8_enlcrop_sc2_mse_shift0_rot0" \
    "$RUNS/supcon_cio_rb_vinc_lat12proj8_enlcrop_sc2_mse_shift0_rot0_nosig" \
    "$RUNS/supcon_cio_rb_vinc_lat12proj8_enlcrop_sc2_mse_shift0_rot0_nosig_warmup100" \
    "$RUNS/supcon_cio_rb_vinc_lat12proj8_enlcrop_sc2_mse_shift0_rot0_warmup100" \
    "$RUNS/supcon_cio_rb_vinc_lat12proj8_enlcrop_sc2_mse_shift1_nojitter" \
    "$RUNS/supcon_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_shift0" \
    "$RUNS/supcon_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_shift0_rot0" \
    "$RUNS/supcon_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_shift0_rot0_lc003" \
    "$RUNS/supcon_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_shift1_nojitter"
do
    echo ""
    echo "--- $(basename $MODEL_DIR) ---"
    $PYTHON scripts/run_cluster_panels.py "$MODEL_DIR" --k 10 --n-panel 16
done

# --- contrastive_cio_vinc (non-RB, lat12) ---
for MODEL_DIR in \
    "$RUNS/contrastive_cio_vinc_lat12proj8_enlcrop_sc2_nl1" \
    "$RUNS/contrastive_cio_vinc_lat12proj8_enlcrop_sc2_nl1_lc100" \
    "$RUNS/contrastive_cio_vinc_lat12proj8_enlcrop_sc2_nl1_lc1e4"
do
    echo ""
    echo "--- $(basename $MODEL_DIR) ---"
    $PYTHON scripts/run_cluster_panels.py "$MODEL_DIR" --k 10 --n-panel 16
done

# --- supcon_cio_vinc (non-RB, lat12) ---
for MODEL_DIR in \
    "$RUNS/supcon_cio_vinc_lat12proj8_enlcrop_sc2_nl1" \
    "$RUNS/supcon_cio_vinc_lat12proj8_enlcrop_sc2_nl1_lc100" \
    "$RUNS/supcon_cio_vinc_lat12proj8_enlcrop_sc2_nl1_lc1e4"
do
    echo ""
    echo "--- $(basename $MODEL_DIR) ---"
    $PYTHON scripts/run_cluster_panels.py "$MODEL_DIR" --k 10 --n-panel 16
done

echo ""
echo "End: $(date)"
