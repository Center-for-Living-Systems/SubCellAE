#!/usr/bin/env bash
#SBATCH --job-name=cnn_le_b1b2
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=logs/slurm/cnn_le_b1b2_%A_%a.out

# Submit as array covering all (fold × budget × repeat) combos for one batch.
# BATCH and JOB_TABLE are passed via --export=ALL,BATCH=b1,JOB_TABLE=...
#
# Simplest usage — run the whole batch at once (non-array):
#   sbatch --export=ALL,BATCH=b1 scripts/sbatch_cnn_le_b1b2.sh
#
# The full LE benchmark for one batch is ~45 jobs (5 folds × 9 budgets × 5 repeats
# for numeric + 5 all-budget runs = 205), but since each job only takes seconds on
# GPU, just run everything serially inside one SLURM job.

set -eo pipefail
exec 2>&1

PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
export PYTHONPATH="$PWD"
mkdir -p logs/slurm

echo "BATCH: ${BATCH}"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Start: $(date)"

$PYTHON scripts/eval_cnn_le_b1b2.py --batch "${BATCH}"

echo "End: $(date)"
