#!/usr/bin/env bash
# Submit one sbatch job per multichannel lc config.
# Run from repo root: bash scripts/submit_conae_cio_vinc_multichannel_lc_sweep.sh

set -eo pipefail

PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
REPO="$PWD"
mkdir -p logs/slurm

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
    NAME=$(basename "$CFG" .yaml | sed 's/ae_contrastive_//')
    JID=$(sbatch \
        --job-name="mc_${NAME:0:20}" \
        --partition=general \
        --gres=gpu:a40:1 \
        --cpus-per-task=8 \
        --mem=64G \
        --time=12:00:00 \
        --output="logs/slurm/mc_lc_${NAME}_%j.out" \
        --wrap="set -e; cd $REPO; export PYTHONPATH=$REPO; $PYTHON scripts/run_ae_from_config.py $CFG" \
        | awk '{print $NF}')
    echo "Submitted $NAME → job $JID"
done
