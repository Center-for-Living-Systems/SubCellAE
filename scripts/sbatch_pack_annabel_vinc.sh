#!/usr/bin/env bash
#SBATCH --job-name=annabel_vinc_pack
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --array=0-8
#SBATCH --output=logs/slurm/annabel_vinc_pack_%A_%a.out

# Pack model.h5 for each of the 9 Annabel vinc result dirs.
# Run after eval (cluster panels + analysis) and cls jobs complete so that
# pack_model_h5.py picks up UMAP, cluster panels, and fa_cls predictions.

set -eo pipefail
exec 2>&1

REPO="$PWD"
PYTHON=/home/liyading/miniconda3/bin/python3
export PYTHONPATH="$REPO:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages"

mkdir -p logs/slurm

DATA_ROOT="/net/projects/CLS/lding/data/fa_data_analysis"

TRAIN_CFGS=($(cat config/annabel_sweep/train_configs.txt))
TRAIN_CFG="${TRAIN_CFGS[$SLURM_ARRAY_TASK_ID]}"

RESULT_NAME=$(grep "result_dir" "$TRAIN_CFG" | sed 's|.*contrastive_run/||; s|".*||')
RESULT_DIR="${DATA_ROOT}/ae_results/contrastive_run/${RESULT_NAME}"

echo "======================================================================"
echo "Job array : $SLURM_ARRAY_JOB_ID  Task: $SLURM_ARRAY_TASK_ID"
echo "Result    : $RESULT_NAME"
echo "[$(date)] Node: $(hostname)"
echo "======================================================================"

$PYTHON scripts/pack_model_h5.py "$RESULT_DIR" --pad-size 64
echo "[$(date)] Done — model.h5 written to $RESULT_DIR"
