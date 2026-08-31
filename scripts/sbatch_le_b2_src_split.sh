#!/usr/bin/env bash
#SBATCH --job-name=le_b2_src
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm/le_b2_src_%A_%a.out

set -eo pipefail
exec 2>&1

REPO="$PWD"
PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
export PYTHONPATH="$REPO"
mkdir -p logs/slurm

RUN_TAG="le_b2_src_split"
JOB_LIST="config/${RUN_TAG}/job_list.txt"
CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$JOB_LIST")
NAME=$(basename "$CONFIG" .yaml)

# Parse variant / fold / repeat from name: le_b2_src_{variant}_fv{f}_r{r}
VARIANT=$(echo "$NAME" | grep -oP '(?<=le_b2_src_)[^_]+(?=_fv)')
FOLD=$(echo "$NAME"    | grep -oP '(?<=_fv)\d+')
REPEAT=$(echo "$NAME"  | grep -oP '(?<=_r)\d+$')

DATA="/net/projects/CLS/lding/data/fa_data_analysis"
ANN_CSV="${DATA}/labelling/${RUN_TAG}/${NAME}.csv"
FOLD_SPLITS="${DATA}/labelling/${RUN_TAG}/fold_splits_ds1.csv"
RUN_DIR="${DATA}/ae_results/contrastive_run/${RUN_TAG}/${NAME}"

echo "task $SLURM_ARRAY_TASK_ID -> $NAME"
echo "variant=$VARIANT  fold=$FOLD  repeat=$REPEAT"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Start: $(date)"

$PYTHON scripts/run_ae_from_config.py "$CONFIG"

echo "--- per-condition LGBM eval ---"
$PYTHON scripts/eval_one_src_split_run.py \
    --run-dir     "$RUN_DIR" \
    --ann-csv     "$ANN_CSV" \
    --fold-splits "$FOLD_SPLITS" \
    --fold        "$FOLD" \
    --variant     "$VARIANT" \
    --repeat      "$REPEAT"

echo "End: $(date)"
