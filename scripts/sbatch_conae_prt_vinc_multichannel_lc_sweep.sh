#!/usr/bin/env bash
#SBATCH --job-name=conae_prt_vinc
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --array=0-17
#SBATCH --output=logs/slurm/conae_prt_vinc_multichannel_%A_%a.out

# ConAE + SupCon vinc — cio_mode_prt — all 18 configs (15 conae + 3 supcon).
# Array index → config file via CONFIGS list below.

set -eo pipefail
exec 2>&1

REPO="$PWD"
PYTHON=/home/liyading/miniconda3/bin/python3
export PYTHONPATH="$REPO:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages"

mkdir -p logs/slurm

CONFIGS=(
    config/contrastive_config/ae_contrastive_prt_vinc_lat12proj8_enlcrop_sc2_nl1.yaml
    config/contrastive_config/ae_contrastive_prt_vinc_lat12proj8_enlcrop_sc2_nl1_lc100.yaml
    config/contrastive_config/ae_contrastive_prt_vinc_lat12proj8_enlcrop_sc2_nl1_lc1e4.yaml
    config/contrastive_config/ae_contrastive_prt_vinc_lat18proj12_enlcrop_sc2_nl1_3ch_pza.yaml
    config/contrastive_config/ae_contrastive_prt_vinc_lat18proj12_enlcrop_sc2_nl1_3ch_pza_lc100.yaml
    config/contrastive_config/ae_contrastive_prt_vinc_lat18proj12_enlcrop_sc2_nl1_3ch_pza_lc1e4.yaml
    config/contrastive_config/ae_contrastive_prt_vinc_lat18proj12_enlcrop_sc2_nl1_4ch_vinc.yaml
    config/contrastive_config/ae_contrastive_prt_vinc_lat18proj12_enlcrop_sc2_nl1_4ch_vinc_lc100.yaml
    config/contrastive_config/ae_contrastive_prt_vinc_lat18proj12_enlcrop_sc2_nl1_4ch_vinc_lc1e4.yaml
    config/contrastive_config/ae_contrastive_prt_vinc_lat24proj16_enlcrop_sc2_nl1_3ch_pza.yaml
    config/contrastive_config/ae_contrastive_prt_vinc_lat24proj16_enlcrop_sc2_nl1_3ch_pza_lc100.yaml
    config/contrastive_config/ae_contrastive_prt_vinc_lat24proj16_enlcrop_sc2_nl1_3ch_pza_lc1e4.yaml
    config/contrastive_config/ae_contrastive_prt_vinc_lat24proj16_enlcrop_sc2_nl1_4ch_vinc.yaml
    config/contrastive_config/ae_contrastive_prt_vinc_lat24proj16_enlcrop_sc2_nl1_4ch_vinc_lc100.yaml
    config/contrastive_config/ae_contrastive_prt_vinc_lat24proj16_enlcrop_sc2_nl1_4ch_vinc_lc1e4.yaml
    config/contrastive_config/ae_supcon_prt_vinc_lat12proj8_enlcrop_sc2_nl1.yaml
    config/contrastive_config/ae_supcon_prt_vinc_lat12proj8_enlcrop_sc2_nl1_lc100.yaml
    config/contrastive_config/ae_supcon_prt_vinc_lat12proj8_enlcrop_sc2_nl1_lc1e4.yaml
)

CFG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

echo "======================================================================"
echo "Job array : $SLURM_ARRAY_JOB_ID  Task: $SLURM_ARRAY_TASK_ID"
echo "Config    : $CFG"
echo "[$(date)] Node: $(hostname)"
echo "======================================================================"

$PYTHON scripts/run_ae_from_config.py "$CFG"
echo "[$(date)] Done"
