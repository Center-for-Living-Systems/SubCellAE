#!/usr/bin/env bash
# submit_translation_ablation.sh
#
# Submit 4 translation-ablation training jobs (ConAE + SupCon, shift=0 and shift=1/nojitter).
# All 4 run in parallel as independent single-GPU jobs.

set -eo pipefail
mkdir -p logs/slurm

REPO="$PWD"
PYTHON=/home/liyading/miniconda3/bin/python3
export PYTHONPATH="$REPO:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages"

CONFIGS=(
  "config/contrastive_config/ae_contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_shift0.yaml"
  "config/contrastive_config/ae_contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_shift1_nojitter.yaml"
  "config/contrastive_config/ae_supcon_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_shift0.yaml"
  "config/contrastive_config/ae_supcon_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_shift1_nojitter.yaml"
)

LABELS=(
  "conae_shift0"
  "conae_shift1_nojitter"
  "supcon_shift0"
  "supcon_shift1_nojitter"
)

echo "Submitting ${#CONFIGS[@]} translation-ablation jobs"
echo ""

for i in "${!CONFIGS[@]}"; do
    CFG="${CONFIGS[$i]}"
    LABEL="${LABELS[$i]}"
    JOB=$(sbatch --parsable \
        --job-name="transl_${LABEL}" \
        --partition=general \
        --gres=gpu:a40:1 \
        --cpus-per-task=8 \
        --mem=32G \
        --time=08:00:00 \
        --output="logs/slurm/transl_${LABEL}_%j.out" \
        --wrap="exec 2>&1
export PYTHONPATH='$PYTHONPATH'
echo 'Config: $CFG'
echo 'Node:   '\$(hostname)
echo 'GPU:    '\$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo 'Start:  '\$(date)
$PYTHON scripts/run_ae_from_config.py '$CFG'
echo 'End:    '\$(date)")
    echo "  [$((i+1))/${#CONFIGS[@]}] ${LABEL}  →  job $JOB"
done

echo ""
echo "Monitor:  squeue -u \$USER"
