#!/usr/bin/env bash
# =============================================================================
# sbatch_screening_jittercrop_vit_hm.sh
#
# One-shot run: ViT-small + jitter crop + histogram matching at train time.
# ViTs benefit from HM (unlike CNNs) because they lack inductive bias toward
# local texture.  Combined with spatial jitter crop this should improve ppax
# generalization over the plain ViT-small sweep_hm baseline.
#
# Submit:
#   sbatch scripts/sbatch_screening_jittercrop_vit_hm.sh
# =============================================================================

#SBATCH --job-name=screen_vit_hm
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/screening_vit_hm_%j.out
#SBATCH --error=logs/screening_vit_hm_%j.err

set -euo pipefail

REPO="/net/projects/CLS/lding/gitcode/SubCellAE-nonad-vs-ad"
cd "$REPO"
mkdir -p logs

PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
echo "Python: $PYTHON"
"$PYTHON" --version

CONFIG="config/screening_config/config_screening_vinc_jittercrop_vit_hm.yaml"
OUT_DIR="/net/projects/CLS/lding/data/fa_data_analysis/ae_results/screening/diversity/jittercrop_vit_hm_sz224"

echo "======================================================"
echo "Job:    ${SLURM_JOB_ID}"
echo "Node:   ${SLURMD_NODENAME}"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Config: ${CONFIG}"
echo "Out:    ${OUT_DIR}"
echo "Start:  $(date)"
echo "======================================================"

$PYTHON scripts/run_screening_from_config.py \
    "$CONFIG" \
    --out_dir "$OUT_DIR"

echo ""
echo "--- ppax generalization eval ---"
$PYTHON scripts/run_screening_eval.py \
    config/screening_config/config_screening_ppax_eval.yaml \
    --model_pt   "${OUT_DIR}/model_best.pt" \
    --backbone   "vit_small_patch16_224" \
    --input_size 224 \
    --out_dir    "${OUT_DIR}/ppax_eval"

echo ""
echo "Done: $(date)"
