#!/usr/bin/env bash
# =============================================================================
# sbatch_screening_sweep.sh
# =============================================================================
# SLURM array job: sweep backbone × input_size for the binary adhesion screening
# classifier.  Each array element trains one (backbone, input_size) combination.
#
# Backbones (6):
#   efficientnet_b0   efficientnet_b2
#   resnet18          resnet50
#   vit_tiny_patch16_224  vit_small_patch16_224
#
# Input sizes (3): 64  128  224
#
# Total jobs: 18  (array indices 0–17)
#
# Submit:
#   sbatch scripts/sbatch_screening_sweep.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/screening_sweep_<JOBID>_<ARRAYID>.out
# =============================================================================

#SBATCH --job-name=screen_sweep
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-17
#SBATCH --output=logs/screening_sweep_%A_%a.out
#SBATCH --error=logs/screening_sweep_%A_%a.err

set -euo pipefail

REPO="/net/projects/CLS/lding/gitcode/SubCellAE-nonad-vs-ad"
cd "$REPO"
mkdir -p logs

set +u
source /home/liyading/miniconda3/etc/profile.d/conda.sh
conda activate subcellae-cuda
set -u

PYTHON=$(which python)

# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------
BACKBONES=(
    "efficientnet_b0"
    "efficientnet_b2"
    "resnet18"
    "resnet50"
    "vit_tiny_patch16_224"
    "vit_small_patch16_224"
)
INPUT_SIZES=(64 128 224)

N_BACKBONES=${#BACKBONES[@]}   # 6
N_SIZES=${#INPUT_SIZES[@]}     # 3

# Map array index → (backbone_idx, size_idx)
BACKBONE_IDX=$(( SLURM_ARRAY_TASK_ID / N_SIZES ))
SIZE_IDX=$(( SLURM_ARRAY_TASK_ID % N_SIZES ))

BACKBONE=${BACKBONES[$BACKBONE_IDX]}
INPUT_SIZE=${INPUT_SIZES[$SIZE_IDX]}

# Sanitise backbone name for directory (replace underscores with hyphens etc.)
BACKBONE_SAFE=$(echo "$BACKBONE" | tr '/' '-')

DATA_ROOT="/net/projects/CLS/lding/data/fa_data_analysis"
OUT_DIR="${DATA_ROOT}/ae_results/screening/sweep/${BACKBONE_SAFE}_sz${INPUT_SIZE}"

echo "======================================================"
echo "Job array:  ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node:       ${SLURMD_NODENAME}"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Backbone:   ${BACKBONE}"
echo "Input size: ${INPUT_SIZE}"
echo "Out dir:    ${OUT_DIR}"
echo "Start:      $(date)"
echo "======================================================"

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
$PYTHON scripts/run_screening_from_config.py \
    config/screening_config/config_screening_vinc.yaml \
    --backbone   "$BACKBONE" \
    --input_size "$INPUT_SIZE" \
    --out_dir    "$OUT_DIR"

# ---------------------------------------------------------------------------
# Eval on ppax (cross-dataset generalization)
# ---------------------------------------------------------------------------
echo ""
echo "--- ppax generalization eval ---"
$PYTHON scripts/run_screening_eval.py \
    config/screening_config/config_screening_ppax_eval.yaml \
    --model_pt   "${OUT_DIR}/model_best.pt" \
    --backbone   "$BACKBONE" \
    --input_size "$INPUT_SIZE" \
    --out_dir    "${OUT_DIR}/ppax_eval"

echo ""
echo "Done: $(date)"
