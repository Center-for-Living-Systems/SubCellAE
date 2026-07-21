#!/usr/bin/env bash
# submit_enlcrop_vs_noenlcrop.sh
#
# Head-to-head: enlcrop(shift0_rot0) vs no-enlcrop
# Both ConAE and SupCon; sc2 (input/2); nl1 loss; lambda_contrast=0.03
#
# Enlcrop:    58×58 context from source frame → center-crop 32×32 at train time
# No-enlcrop: pre-extracted 32×32 from tiff_patches32_mr10 (same coordinates)
# Both use patch/input_divisor=2.0, so pixel values are identical / 2

set -eo pipefail
mkdir -p logs/slurm

REPO="$PWD"
PYTHON=/home/liyading/miniconda3/bin/python3
export PYTHONPATH="$REPO:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages"

CONFIGS=(
  "config/contrastive_config/ae_contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_shift0_rot0_lc003.yaml"
  "config/contrastive_config/ae_contrastive_cio_rb_vinc_lat12proj8_noenlcrop_sc2_nl1_lc003.yaml"
  "config/contrastive_config/ae_supcon_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_shift0_rot0_lc003.yaml"
  "config/contrastive_config/ae_supcon_cio_rb_vinc_lat12proj8_noenlcrop_sc2_nl1_lc003.yaml"
)

LABELS=(
  "conae_enlcrop_nl1_lc003"
  "conae_noenlcrop_nl1_lc003"
  "supcon_enlcrop_nl1_lc003"
  "supcon_noenlcrop_nl1_lc003"
)

echo "Submitting ${#CONFIGS[@]} enlcrop-vs-noenlcrop comparison jobs"
echo ""

for i in "${!CONFIGS[@]}"; do
    CFG="${CONFIGS[$i]}"
    LABEL="${LABELS[$i]}"
    JOB=$(sbatch --parsable \
        --job-name="${LABEL}" \
        --partition=general \
        --gres=gpu:a40:1 \
        --cpus-per-task=8 \
        --mem=32G \
        --time=08:00:00 \
        --output="logs/slurm/${LABEL}_%j.out" \
        --wrap="exec 2>&1
export PYTHONPATH='$PYTHONPATH'
echo 'Config: $CFG'
echo 'Node:   '\$(hostname)
echo 'GPU:    '\$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo 'Start:  '\$(date)
$PYTHON scripts/run_ae_from_config.py '$CFG'
echo 'End:    '\$(date)")
    echo "  [$((i+1))/${#CONFIGS[@]}] ${LABEL}  ->  job $JOB"
done

echo ""
echo "Monitor:  squeue -u \$USER"
