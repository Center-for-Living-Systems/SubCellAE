#!/usr/bin/env bash
#SBATCH --job-name=combo_clip02_train
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-14
#SBATCH --output=logs/slurm/ds_combo_sc2_clip02_l1_train_%A_%a.out

# Training sweep — clip [0,2] + sc2 (÷2) / L1 — 15 dataset combinations.
# CIO patches (not CIO-RB).

set -eo pipefail
exec 2>&1

REPO="$PWD"
export PYTHONPATH="$REPO:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages"
PYTHON=/home/liyading/miniconda3/bin/python3

mkdir -p logs/slurm

COMBO_LIST="config/contrastive_config/ds_combo_v3/combo_list.txt"
COMBO=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$COMBO_LIST")
CFG="config/contrastive_config/ds_combo_v3/ae_conae_enlcrop_sc2_clip02_l1_lc010_bal_${COMBO}.yaml"

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
