#!/usr/bin/env bash
# submit_logmap_route.sh
#
# Log-map route: ConAE + SupCon with shifted-log preprocessing
# No enlcrop; sc2 (patch_input_divisor=2.0); nl1 loss; lambda_contrast=0.03
# Logs both training loss (log-mapped space) and orig_L1 (CIO-RB space)
#
# Comparable to jobs 1018874–1018877 (enlcrop vs noenlcrop comparison)

set -eo pipefail
mkdir -p logs/slurm

REPO="$PWD"
PYTHON=/home/liyading/miniconda3/bin/python3
export PYTHONPATH="$REPO:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages"

CONFIGS=(
  "config/contrastive_config/ae_contrastive_cio_rb_vinc_lat12proj8_noenlcrop_sc2_nl1_lc003_logmap.yaml"
  "config/contrastive_config/ae_supcon_cio_rb_vinc_lat12proj8_noenlcrop_sc2_nl1_lc003_logmap.yaml"
)

LABELS=(
  "conae_noenlcrop_nl1_lc003_logmap"
  "supcon_noenlcrop_nl1_lc003_logmap"
)

echo "Submitting ${#CONFIGS[@]} log-map route jobs"
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
