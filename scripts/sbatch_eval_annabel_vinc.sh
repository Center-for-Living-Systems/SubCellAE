#!/usr/bin/env bash
#SBATCH --job-name=annabel_vinc_eval
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=0-8
#SBATCH --output=logs/slurm/annabel_vinc_eval_%A_%a.out

# Annabel vinc control sweep — eval per result dir:
#   cluster panels k=3,6,10 | violin plots | analysis (UMAP + clustering)

set -eo pipefail
exec 2>&1

REPO="$PWD"
PYTHON=/home/liyading/miniconda3/bin/python3
export PYTHONPATH="$REPO:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages"

mkdir -p logs/slurm

DATA_ROOT="/net/projects/CLS/lding/data/fa_data_analysis"

TRAIN_CFGS=($(cat config/annabel_sweep/train_configs.txt))
ANALYSIS_CFGS=($(cat config/annabel_sweep/analysis_configs.txt))

TRAIN_CFG="${TRAIN_CFGS[$SLURM_ARRAY_TASK_ID]}"
ANALYSIS_CFG="${ANALYSIS_CFGS[$SLURM_ARRAY_TASK_ID]}"

# Extract result_dir from the training config result_dir field
RESULT_NAME=$(grep "result_dir" "$TRAIN_CFG" | sed 's|.*contrastive_run/||; s|".*||')
RESULT_DIR="${DATA_ROOT}/ae_results/contrastive_run/${RESULT_NAME}"

echo "======================================================================"
echo "Job array  : $SLURM_ARRAY_JOB_ID  Task: $SLURM_ARRAY_TASK_ID"
echo "Result dir : $RESULT_DIR"
echo "[$(date)] Node: $(hostname)"
echo "======================================================================"

echo "--- KMeans cluster panels k=3 ---"
$PYTHON scripts/run_cluster_panels.py "$RESULT_DIR" --k 3  --out-dir "$RESULT_DIR/eval/cluster_panels_k3"

echo "--- KMeans cluster panels k=6 ---"
$PYTHON scripts/run_cluster_panels.py "$RESULT_DIR" --k 6  --out-dir "$RESULT_DIR/eval/cluster_panels_k6"

echo "--- KMeans cluster panels k=10 ---"
$PYTHON scripts/run_cluster_panels.py "$RESULT_DIR" --k 10 --out-dir "$RESULT_DIR/eval/cluster_panels_k10"

echo "--- Violin plots ---"
$PYTHON scripts/plot_recon_metric_violins.py "$RESULT_DIR"

echo "--- Analysis (UMAP + clustering) ---"
$PYTHON scripts/run_analysis_from_config.py "$ANALYSIS_CFG"

echo "[$(date)] Done"
