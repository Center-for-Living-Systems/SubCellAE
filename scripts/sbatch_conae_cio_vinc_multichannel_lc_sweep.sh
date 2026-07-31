#!/usr/bin/env bash
#SBATCH --job-name=conae_mc_lc
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/conae_cio_vinc_multichannel_lc_%j.out

# ConAE: CIO vinc multichannel — retrain base (4ch/3ch lat18/lat24) + lc sweep
# Runs all 12 multichannel configs sequentially on one GPU.
# Configs:
#   lat18 3ch_pza: lc=0.5 (retrain), lc=100, lc=0.0001
#   lat18 4ch_vinc: lc=0.5 (retrain), lc=100, lc=0.0001
#   lat24 3ch_pza: lc=0.5 (retrain), lc=100, lc=0.0001
#   lat24 4ch_vinc: lc=0.5 (retrain), lc=100, lc=0.0001

set -eo pipefail
exec 2>&1

REPO="$PWD"
PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
export PYTHONPATH="$REPO"
mkdir -p logs/slurm

echo "======================================================================"
echo "Job:   $SLURM_JOB_ID"
echo "Node:  $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Start: $(date)"
echo "======================================================================"

CONFIGS=(
    config/contrastive_config/ae_contrastive_cio_vinc_lat18proj12_enlcrop_sc2_nl1_3ch_pza.yaml
    config/contrastive_config/ae_contrastive_cio_vinc_lat18proj12_enlcrop_sc2_nl1_3ch_pza_lc100.yaml
    config/contrastive_config/ae_contrastive_cio_vinc_lat18proj12_enlcrop_sc2_nl1_3ch_pza_lc1e4.yaml
    config/contrastive_config/ae_contrastive_cio_vinc_lat18proj12_enlcrop_sc2_nl1_4ch_vinc.yaml
    config/contrastive_config/ae_contrastive_cio_vinc_lat18proj12_enlcrop_sc2_nl1_4ch_vinc_lc100.yaml
    config/contrastive_config/ae_contrastive_cio_vinc_lat18proj12_enlcrop_sc2_nl1_4ch_vinc_lc1e4.yaml
    config/contrastive_config/ae_contrastive_cio_vinc_lat24proj16_enlcrop_sc2_nl1_3ch_pza.yaml
    config/contrastive_config/ae_contrastive_cio_vinc_lat24proj16_enlcrop_sc2_nl1_3ch_pza_lc100.yaml
    config/contrastive_config/ae_contrastive_cio_vinc_lat24proj16_enlcrop_sc2_nl1_3ch_pza_lc1e4.yaml
    config/contrastive_config/ae_contrastive_cio_vinc_lat24proj16_enlcrop_sc2_nl1_4ch_vinc.yaml
    config/contrastive_config/ae_contrastive_cio_vinc_lat24proj16_enlcrop_sc2_nl1_4ch_vinc_lc100.yaml
    config/contrastive_config/ae_contrastive_cio_vinc_lat24proj16_enlcrop_sc2_nl1_4ch_vinc_lc1e4.yaml
)

for CFG in "${CONFIGS[@]}"; do
    echo ""
    echo "--- $(basename $CFG .yaml) ---"
    echo "Start: $(date)"
    $PYTHON scripts/run_ae_from_config.py "$CFG"
    echo "Done:  $(date)"
done

echo ""
echo "All configs complete. End: $(date)"
