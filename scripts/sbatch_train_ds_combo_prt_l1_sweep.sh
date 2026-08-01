#!/usr/bin/env bash
#SBATCH --job-name=combo_prt_train
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-14
#SBATCH --output=logs/slurm/ds_combo_prt_l1_train_%A_%a.out

# Training sweep — cio_mode_prt / no clip / no sigmoid / L1 — 15 dataset combos.

set -eo pipefail
exec 2>&1

REPO="$PWD"
export PYTHONPATH="$REPO:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages"
PYTHON=/home/liyading/miniconda3/bin/python3

mkdir -p logs/slurm

COMBO_LIST="config/contrastive_config/ds_combo_v4/combo_list.txt"
COMBO=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$COMBO_LIST")
CFG="config/contrastive_config/ds_combo_v4/ae_conae_enlcrop_prt_l1_lc010_bal_${COMBO}.yaml"

echo "======================================================================"
echo "Job array : $SLURM_ARRAY_JOB_ID  Task: $SLURM_ARRAY_TASK_ID"
echo "Combo     : $COMBO"
echo "Config    : $CFG"
echo "[$(date)] Node: $(hostname)"
echo "======================================================================"

$PYTHON scripts/run_ae_from_config.py "$CFG"
echo "[$(date)] Done"
