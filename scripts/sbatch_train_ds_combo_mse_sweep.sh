#!/usr/bin/env bash
#SBATCH --job-name=ds_combo_mse_train
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-14
#SBATCH --output=logs/slurm/ds_combo_mse_train_%A_%a.out

# Training sweep — MSE loss — 15 dataset combinations, enlcrop/sc2/lc010/balanced.

set -eo pipefail
exec 2>&1

REPO="$PWD"
export PYTHONPATH="$REPO:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages"
PYTHON=/home/liyading/miniconda3/bin/python3

mkdir -p logs/slurm

COMBO_LIST="config/contrastive_config/ds_combo_v2/combo_list.txt"
COMBO=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$COMBO_LIST")
CFG="config/contrastive_config/ds_combo_v2/ae_conae_enlcrop_sc2_mse_lc010_bal_${COMBO}.yaml"

echo "======================================================================"
echo "Job array : $SLURM_ARRAY_JOB_ID  Task: $SLURM_ARRAY_TASK_ID"
echo "Combo     : $COMBO"
echo "Config    : $CFG"
echo "Node      : $(hostname)"
echo "GPU       : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Start     : $(date)"
echo "======================================================================"

$PYTHON scripts/run_ae_from_config.py "$CFG"

echo "End: $(date)"
