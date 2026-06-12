#!/usr/bin/env bash
# =============================================================================
# sbatch_screening_sweep_hm.sh
# =============================================================================
# Same 6 × 3 backbone/size sweep as sbatch_screening_sweep.sh but with
# histogram-matching normalisation applied to training patches (--pixel_correction histogram).
# Each job:
#   1. Trains on vinc with HM normalisation (saves reference CDF to out_dir)
#   2. Evaluates val set
#   3. Evaluates ppax using the saved reference CDF for consistent HM correction
#
# Submit: sbatch scripts/sbatch_screening_sweep_hm.sh
# =============================================================================

#SBATCH --job-name=screen_sweep_hm
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-17
#SBATCH --output=logs/screening_sweep_hm_%A_%a.out
#SBATCH --error=logs/screening_sweep_hm_%A_%a.err

set -euo pipefail

REPO="/net/projects/CLS/lding/gitcode/SubCellAE-nonad-vs-ad"
cd "$REPO"
mkdir -p logs

set +u
source /home/liyading/miniconda3/etc/profile.d/conda.sh
conda activate subcellae-cuda
set -u

PYTHON=$(which python)

BACKBONES=(
    "efficientnet_b0"
    "efficientnet_b2"
    "resnet18"
    "resnet50"
    "vit_tiny_patch16_224"
    "vit_small_patch16_224"
)
INPUT_SIZES=(64 128 224)

N_BACKBONES=${#BACKBONES[@]}
N_SIZES=${#INPUT_SIZES[@]}

BACKBONE_IDX=$(( SLURM_ARRAY_TASK_ID / N_SIZES ))
SIZE_IDX=$(( SLURM_ARRAY_TASK_ID % N_SIZES ))

BACKBONE=${BACKBONES[$BACKBONE_IDX]}
INPUT_SIZE=${INPUT_SIZES[$SIZE_IDX]}
BACKBONE_SAFE=$(echo "$BACKBONE" | tr '/' '-')

DATA_ROOT="/net/projects/CLS/lding/data/fa_data_analysis"
OUT_DIR="${DATA_ROOT}/ae_results/screening/sweep_hm/${BACKBONE_SAFE}_sz${INPUT_SIZE}"

echo "======================================================"
echo "Job:        ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node:       ${SLURMD_NODENAME}"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Backbone:   ${BACKBONE}"
echo "Input size: ${INPUT_SIZE}"
echo "Correction: histogram"
echo "Out dir:    ${OUT_DIR}"
echo "Start:      $(date)"
echo "======================================================"

# ── 1. Train with histogram matching ─────────────────────────────────────────
$PYTHON scripts/run_screening_from_config.py \
    config/screening_config/config_screening_vinc_hm.yaml \
    --backbone        "$BACKBONE" \
    --input_size      "$INPUT_SIZE" \
    --pixel_correction histogram \
    --out_dir         "$OUT_DIR"

# ── 2. Eval on ppax — reuse the reference CDF saved during training ───────────
echo ""
echo "--- ppax generalization eval (histogram matching, saved CDF) ---"
$PYTHON scripts/run_screening_eval.py \
    config/screening_config/config_screening_ppax_eval.yaml \
    --model_pt   "${OUT_DIR}/model_best.pt" \
    --backbone   "$BACKBONE" \
    --input_size "$INPUT_SIZE" \
    --out_dir    "${OUT_DIR}/ppax_eval"

echo ""
echo "Done: $(date)"
