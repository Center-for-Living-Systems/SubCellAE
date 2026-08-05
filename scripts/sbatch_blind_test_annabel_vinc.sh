#!/usr/bin/env bash
#SBATCH --job-name=annabel_blind_test
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --array=0-8
#SBATCH --output=logs/slurm/annabel_blind_test_%A_%a.out

# Blind cross-dataset evaluation: apply 9 Annabel-vinc AE+LightGBM models
# to vinc (control+ycomp), ppax (control), pfak (control).
# Evaluate against Margaret's independent label CSVs (labels_*_20260521.csv).

set -eo pipefail
exec 2>&1

REPO="$PWD"
PYTHON=/home/liyading/miniconda3/bin/python3
export PYTHONPATH="$REPO:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages"

mkdir -p logs/slurm

DATA_ROOT="/net/projects/CLS/lding/data/fa_data_analysis"
TRAIN_CFGS=($(cat config/annabel_sweep/train_configs.txt))
TRAIN_CFG="${TRAIN_CFGS[$SLURM_ARRAY_TASK_ID]}"

# Extract result name from training config
RESULT_NAME=$(grep "result_dir" "$TRAIN_CFG" | sed 's|.*contrastive_run/||; s|".*||')
RESULT_DIR="${DATA_ROOT}/ae_results/contrastive_run/${RESULT_NAME}"

echo "======================================================================"
echo "Job array  : $SLURM_ARRAY_JOB_ID  Task: $SLURM_ARRAY_TASK_ID"
echo "Result dir : $RESULT_DIR"
echo "[$(date)] Node: $(hostname)"
echo "======================================================================"

$PYTHON scripts/run_blind_test_crossds.py "$RESULT_DIR" --device cuda

echo "[$(date)] Done"
