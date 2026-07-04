#!/usr/bin/env bash
#SBATCH --job-name=cross_ds_eval_enlcrop
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/cross_ds_eval_enlcrop_%j.out

# Cross-dataset reconstruction violin plots (MSE / L1 / Hessian-L1) for
# all enlcrop_sc2 contrastive + supcon models that are missing these plots.
# External datasets evaluated: pfak, ppax, nih3t3 (control + ycomp each).
#
# Output per model: <model_dir>/._cross_dataset_recon_{mse,l1,hessian_l1}.png
#                   <model_dir>/cross_dataset_recon_metrics.csv

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

# --- contrastive (ConAE) enlcrop_sc2 vinc-only — missing plots ---
for MODEL_DIR in \
    "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2" \
    "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_l1" \
    "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_lr4" \
    "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_lr8" \
    "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1" \
    "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_lc0125" \
    "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_lc025" \
    "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nmse" \
    "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nmse_lc0125" \
    "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nmse_lc025"
do
    echo ""
    echo "--- cross_dataset_eval: $(basename $MODEL_DIR) ---"
    $PYTHON scripts/run_cross_dataset_eval.py "$MODEL_DIR" \
        --mode sweep --root-folder "$ROOT"
done

# --- supervised contrastive (SupConAE) enlcrop_sc2 vinc-only — missing plots ---
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
    echo "--- cross_dataset_eval: $(basename $MODEL_DIR) ---"
    $PYTHON scripts/run_cross_dataset_eval.py "$MODEL_DIR" \
        --mode sweep --root-folder "$ROOT"
done

echo ""
echo "End: $(date)"
