#!/usr/bin/env bash
# submit_mse_ablation.sh
#
# MSE-loss versions of the translation/rotation ablation runs.
# 6 jobs: ConAE + SupCon × shift0 / shift1_nojitter / shift0_rot0

set -eo pipefail
mkdir -p logs/slurm

REPO="$PWD"
PYTHON=/home/liyading/miniconda3/bin/python3
export PYTHONPATH="$REPO:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages"

CONFIGS=(
  "config/contrastive_config/ae_contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_mse_shift0.yaml"
  "config/contrastive_config/ae_contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_mse_shift1_nojitter.yaml"
  "config/contrastive_config/ae_contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_mse_shift0_rot0.yaml"
  "config/contrastive_config/ae_supcon_cio_rb_vinc_lat12proj8_enlcrop_sc2_mse_shift0.yaml"
  "config/contrastive_config/ae_supcon_cio_rb_vinc_lat12proj8_enlcrop_sc2_mse_shift1_nojitter.yaml"
  "config/contrastive_config/ae_supcon_cio_rb_vinc_lat12proj8_enlcrop_sc2_mse_shift0_rot0.yaml"
)

LABELS=(
  "conae_mse_shift0"
  "conae_mse_shift1_nojitter"
  "conae_mse_shift0_rot0"
  "supcon_mse_shift0"
  "supcon_mse_shift1_nojitter"
  "supcon_mse_shift0_rot0"
)

echo "Submitting ${#CONFIGS[@]} MSE ablation jobs"
echo ""

for i in "${!CONFIGS[@]}"; do
    CFG="${CONFIGS[$i]}"
    LABEL="${LABELS[$i]}"
    JOB=$(sbatch --parsable \
        --job-name="mse_${LABEL}" \
        --partition=general \
        --gres=gpu:a40:1 \
        --cpus-per-task=8 \
        --mem=32G \
        --time=08:00:00 \
        --output="logs/slurm/mse_${LABEL}_%j.out" \
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
