#!/usr/bin/env bash
#SBATCH --job-name=le_b1b2_le
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm/le_b1b2_le_%A_%a.out

set -eo pipefail
exec 2>&1

REPO="$PWD"
PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
export PYTHONPATH="$REPO"
mkdir -p logs/slurm

# RUN_TAG passed via --export=RUN_TAG=... at submission time
JOB_LIST="config/${RUN_TAG}/job_list.txt"
CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$JOB_LIST")
NAME=$(basename "$CONFIG" .yaml)

FOLD=$(echo "$NAME"   | grep -oP '(?<=_fv)\d+')
BUDGET=$(echo "$NAME" | grep -oP '(?<=_nb)[^_]+')
REPEAT=$(echo "$NAME" | grep -oP '(?<=_r)\d+$')

# Derive batch (b1 or b2) from run tag
BATCH=$(echo "$RUN_TAG" | grep -oP '(?<=b1b2_)\w+(?=_lat)')

DATA="/net/projects/CLS/lding/data/fa_data_analysis"
ANN_CSV="${DATA}/labelling/le_b1b2_le/le_b1b2_${BATCH}_fv${FOLD}_nb${BUDGET}_r${REPEAT}.csv"
FOLD_SPLITS="${DATA}/labelling/le_b1b2_matched/fold_splits_${BATCH}.csv"
RUN_DIR="${DATA}/ae_results/contrastive_run/${RUN_TAG}/${NAME}"

echo "task $SLURM_ARRAY_TASK_ID -> $NAME  (batch=$BATCH)"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Start: $(date)"

$PYTHON scripts/run_ae_from_config.py "$CONFIG"

echo "--- LGBM eval ---"
$PYTHON scripts/eval_one_supcon_run.py \
    --run-dir     "$RUN_DIR" \
    --ann-csv     "$ANN_CSV" \
    --fold-splits "$FOLD_SPLITS" \
    --fold        "$FOLD" \
    --budget      "$BUDGET" \
    --repeat      "$REPEAT"

echo "End: $(date)"
