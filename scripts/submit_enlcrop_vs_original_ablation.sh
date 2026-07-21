#!/usr/bin/env bash
# submit_enlcrop_vs_original_ablation.sh
#
# Full 2x2x2 ablation over (output_sigmoid, sc2, warmup) for both ConAE and SupCon,
# with shift0_rot0 (no spatial aug) and MSE loss.
# Goal: isolate which of these settings causes reconstruction difference
#       vs the original supcon_cio_rb_vinc_lat12proj8.
#
# Already running (submitted earlier, not re-submitted here):
#   enlcrop_sc2_mse_shift0_rot0  (sigmoid=T, sc2=T, warmup=0)  jobs 1017608, 1017611
#
# Combination key:
#   sigmoid : output_sigmoid true/false
#   sc2     : input_divisor 2.0 / 1.0
#   warmup  : warmup_epochs 0 / 100
#
# Original supcon_cio_rb_vinc_lat12proj8 = (nosig, nosc2, warmup100)

set -eo pipefail
mkdir -p logs/slurm

REPO="$PWD"
PYTHON=/home/liyading/miniconda3/bin/python3
export PYTHONPATH="$REPO:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages"

# ── SupCon ────────────────────────────────────────────────────────────────────
SC_CONFIGS=(
  # sigmoid=T, sc2=T, warmup=100
  "config/contrastive_config/ae_supcon_cio_rb_vinc_lat12proj8_enlcrop_sc2_mse_shift0_rot0_warmup100.yaml"
  # sigmoid=T, sc2=F, warmup=0
  "config/contrastive_config/ae_supcon_cio_rb_vinc_lat12proj8_enlcrop_mse_shift0_rot0.yaml"
  # sigmoid=T, sc2=F, warmup=100
  "config/contrastive_config/ae_supcon_cio_rb_vinc_lat12proj8_enlcrop_mse_shift0_rot0_warmup100.yaml"
  # sigmoid=F, sc2=T, warmup=0
  "config/contrastive_config/ae_supcon_cio_rb_vinc_lat12proj8_enlcrop_sc2_mse_shift0_rot0_nosig.yaml"
  # sigmoid=F, sc2=T, warmup=100
  "config/contrastive_config/ae_supcon_cio_rb_vinc_lat12proj8_enlcrop_sc2_mse_shift0_rot0_nosig_warmup100.yaml"
  # sigmoid=F, sc2=F, warmup=0
  "config/contrastive_config/ae_supcon_cio_rb_vinc_lat12proj8_enlcrop_mse_shift0_rot0_nosig.yaml"
  # sigmoid=F, sc2=F, warmup=100  <-- closest to original
  "config/contrastive_config/ae_supcon_cio_rb_vinc_lat12proj8_enlcrop_mse_shift0_rot0_nosig_warmup100.yaml"
)
SC_LABELS=(
  "sc_sig_sc2_w100"
  "sc_sig_nosc2_w0"
  "sc_sig_nosc2_w100"
  "sc_nosig_sc2_w0"
  "sc_nosig_sc2_w100"
  "sc_nosig_nosc2_w0"
  "sc_nosig_nosc2_w100"
)

# ── ConAE ─────────────────────────────────────────────────────────────────────
CT_CONFIGS=(
  "config/contrastive_config/ae_contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_mse_shift0_rot0_warmup100.yaml"
  "config/contrastive_config/ae_contrastive_cio_rb_vinc_lat12proj8_enlcrop_mse_shift0_rot0.yaml"
  "config/contrastive_config/ae_contrastive_cio_rb_vinc_lat12proj8_enlcrop_mse_shift0_rot0_warmup100.yaml"
  "config/contrastive_config/ae_contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_mse_shift0_rot0_nosig.yaml"
  "config/contrastive_config/ae_contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_mse_shift0_rot0_nosig_warmup100.yaml"
  "config/contrastive_config/ae_contrastive_cio_rb_vinc_lat12proj8_enlcrop_mse_shift0_rot0_nosig.yaml"
  "config/contrastive_config/ae_contrastive_cio_rb_vinc_lat12proj8_enlcrop_mse_shift0_rot0_nosig_warmup100.yaml"
)
CT_LABELS=(
  "ct_sig_sc2_w100"
  "ct_sig_nosc2_w0"
  "ct_sig_nosc2_w100"
  "ct_nosig_sc2_w0"
  "ct_nosig_sc2_w100"
  "ct_nosig_nosc2_w0"
  "ct_nosig_nosc2_w100"
)

echo "Submitting 14 enlcrop-vs-original ablation jobs"
echo "(sigmoid x sc2 x warmup) x (ConAE + SupCon), shift0_rot0, MSE"
echo ""

submit_group() {
    local CONFIGS=("${!1}")
    local LABELS=("${!2}")
    for i in "${!CONFIGS[@]}"; do
        CFG="${CONFIGS[$i]}"
        LABEL="${LABELS[$i]}"
        JOB=$(sbatch --parsable \
            --job-name="abl_${LABEL}" \
            --partition=general \
            --gres=gpu:a40:1 \
            --cpus-per-task=8 \
            --mem=32G \
            --time=08:00:00 \
            --output="logs/slurm/abl_${LABEL}_%j.out" \
            --wrap="exec 2>&1
export PYTHONPATH='$PYTHONPATH'
echo 'Config: $CFG'
echo 'Node:   '\$(hostname)
echo 'GPU:    '\$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo 'Start:  '\$(date)
$PYTHON scripts/run_ae_from_config.py '$CFG'
echo 'End:    '\$(date)")
        echo "  ${LABEL}  ->  job $JOB"
    done
}

echo "--- SupCon ---"
submit_group SC_CONFIGS[@] SC_LABELS[@]
echo ""
echo "--- ConAE ---"
submit_group CT_CONFIGS[@] CT_LABELS[@]

echo ""
echo "Monitor:  squeue -u \$USER"
echo ""
echo "Design matrix (sigmoid | sc2 | warmup):"
echo "  Already running:  T | T | 0   (jobs 1017608 ConAE, 1017611 SupCon)"
echo "  sc_sig_sc2_w100:  T | T | 100"
echo "  sc_sig_nosc2_w0:  T | F | 0"
echo "  sc_sig_nosc2_w100:T | F | 100"
echo "  sc_nosig_sc2_w0:  F | T | 0"
echo "  sc_nosig_sc2_w100:F | T | 100"
echo "  sc_nosig_nosc2_w0:F | F | 0"
echo "  sc_nosig_nosc2_w100: F | F | 100  <- closest to original"
