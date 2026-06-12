#!/usr/bin/env bash
# =============================================================================
# sbatch_screening_diversity_sweep.sh
# =============================================================================
# 8 training conditions × 2 representative backbones = 16 array jobs
#
#  Conditions:
#   0  baseline               — pax ch1 only,  no jitter
#   1  jitter                 — pax ch1 only,  +intensity jitter
#   2  multichannel           — pax ch1 + vinc ch0, no jitter
#   3  jitter+mc              — pax ch1 + vinc ch0, +intensity jitter
#   4  jittercrop             — pax ch1, +intensity jitter + spatial jitter crop
#                               (±4px translation, ±15° rotation from source frames)
#   5  jittercrop+mc          — pax ch1 jitter crop + vinc ch0 static + intensity jitter
#   6  jittercrop+gamma       — jitter crop + intensity jitter + gamma [0.4, 2.5]
#   7  jittercrop+gamma_mild  — jitter crop + intensity jitter + gamma [0.7, 1.5]
#
#  Backbones (best val / best ppax from first sweep):
#   A  efficientnet_b0   224px  ← best val accuracy
#   B  efficientnet_b2   224px  ← best ppax generalization
#
#  Array index: 0-15  → (condition_idx × 2 + backbone_idx)
#
# NOTE: Conditions 4-7 require source frames (SLURM job 933033, completed):
#   ae_results/source_frames/cio_rb/vinc/{control,ycomp}/
#
# Submit:
#   sbatch scripts/sbatch_screening_diversity_sweep.sh
#   To run only the new condition: --array=14-15
# =============================================================================

#SBATCH --job-name=screen_diversity
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-15
#SBATCH --output=logs/screening_diversity_%A_%a.out
#SBATCH --error=logs/screening_diversity_%A_%a.err

set -euo pipefail

REPO="/net/projects/CLS/lding/gitcode/SubCellAE-nonad-vs-ad"
cd "$REPO"
mkdir -p logs

PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
echo "Python: $PYTHON"
"$PYTHON" --version

# ── grid ──────────────────────────────────────────────────────────────────────
CONFIGS=(
    "config/screening_config/config_screening_vinc.yaml"                          # 0 baseline
    "config/screening_config/config_screening_vinc_jitter.yaml"                   # 1 jitter
    "config/screening_config/config_screening_vinc_multichannel.yaml"             # 2 multichannel
    "config/screening_config/config_screening_vinc_jitter_multichannel.yaml"      # 3 jitter+mc
    "config/screening_config/config_screening_vinc_jittercrop.yaml"               # 4 jitter crop
    "config/screening_config/config_screening_vinc_jittercrop_multichannel.yaml"  # 5 jittercrop+mc
    "config/screening_config/config_screening_vinc_jittercrop_gamma.yaml"          # 6 jittercrop+gamma
    "config/screening_config/config_screening_vinc_jittercrop_gamma_mild.yaml"    # 7 jittercrop+gamma_mild
)
COND_NAMES=(
    "baseline"
    "jitter"
    "multichannel"
    "jitter_mc"
    "jittercrop"
    "jittercrop_mc"
    "jittercrop_gamma"
    "jittercrop_gamma_mild"
)
BACKBONES=("efficientnet_b0" "efficientnet_b2")
INPUT_SIZE=224
N_BACKBONES=${#BACKBONES[@]}   # 2

COND_IDX=$(( SLURM_ARRAY_TASK_ID / N_BACKBONES ))
BB_IDX=$(( SLURM_ARRAY_TASK_ID % N_BACKBONES ))

CONFIG=${CONFIGS[$COND_IDX]}
COND_NAME=${COND_NAMES[$COND_IDX]}
BACKBONE=${BACKBONES[$BB_IDX]}

DATA_ROOT="/net/projects/CLS/lding/data/fa_data_analysis"
OUT_DIR="${DATA_ROOT}/ae_results/screening/diversity/${COND_NAME}_${BACKBONE}_sz${INPUT_SIZE}"

echo "======================================================"
echo "Job:       ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node:      ${SLURMD_NODENAME}"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Condition: ${COND_NAME}"
echo "Backbone:  ${BACKBONE}"
echo "Config:    ${CONFIG}"
echo "Out dir:   ${OUT_DIR}"
echo "Start:     $(date)"
echo "======================================================"

# ── train ─────────────────────────────────────────────────────────────────────
$PYTHON scripts/run_screening_from_config.py \
    "$CONFIG" \
    --backbone   "$BACKBONE" \
    --input_size "$INPUT_SIZE" \
    --out_dir    "$OUT_DIR"

# ── ppax eval (histogram-matched) ─────────────────────────────────────────────
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
