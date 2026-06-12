#!/usr/bin/env bash
# =============================================================================
# sbatch_screening_gradcam.sh
#
# Generate Grad-CAM visualisations for all runs in the diversity sweep.
# Each run produces {run_dir}/gradcam/{train,val}/{patch_name}.png
#
# Runs on a CPU node (no GPU needed — model is small and batches are size 1).
#
# Submit:
#   sbatch scripts/sbatch_screening_gradcam.sh
#
# To run a single model interactively:
#   PYTHON=... python scripts/run_screening_gradcam.py \
#       --model_dir /path/to/run/dir [--device cpu]
# =============================================================================

#SBATCH --job-name=screen_gradcam
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/screening_gradcam_%A_%a.out
#SBATCH --error=logs/screening_gradcam_%A_%a.err

set -euo pipefail

REPO="/net/projects/CLS/lding/gitcode/SubCellAE-nonad-vs-ad"
cd "$REPO"
mkdir -p logs

set +u
source /home/liyading/miniconda3/etc/profile.d/conda.sh
conda activate subcellae-cuda
set -u

PYTHON=$(which python)

SWEEP_ROOT="/net/projects/CLS/lding/data/fa_data_analysis/ae_results/screening/diversity"

MODEL_DIRS=(
    "$SWEEP_ROOT/baseline_efficientnet_b0_sz224"
    "$SWEEP_ROOT/baseline_efficientnet_b2_sz224"
    "$SWEEP_ROOT/jitter_efficientnet_b0_sz224"
    "$SWEEP_ROOT/jitter_efficientnet_b2_sz224"
    "$SWEEP_ROOT/multichannel_efficientnet_b0_sz224"
    "$SWEEP_ROOT/multichannel_efficientnet_b2_sz224"
    "$SWEEP_ROOT/jitter_mc_efficientnet_b0_sz224"
    "$SWEEP_ROOT/jitter_mc_efficientnet_b2_sz224"
)

MODEL_DIR="${MODEL_DIRS[$SLURM_ARRAY_TASK_ID]}"

echo "Job ID:     $SLURM_JOB_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Model dir:  $MODEL_DIR"
echo "Start:      $(date)"
echo ""

$PYTHON -u scripts/run_screening_gradcam.py \
    --model_dir "$MODEL_DIR" \
    --device cpu \
    --log_level INFO

echo ""
echo "Done: $(date)"
