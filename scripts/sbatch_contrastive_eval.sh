#!/usr/bin/env bash
#SBATCH --job-name=con_eval
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=logs/slurm/con_eval_%j.out

# Usage: sbatch scripts/sbatch_contrastive_eval.sh <result_dir>
#   result_dir: full path to a contrastive_run subdirectory containing latents.csv

set -eo pipefail
exec 2>&1

RESULT_DIR="${1:?Usage: sbatch sbatch_contrastive_eval.sh <result_dir>}"

REPO="$PWD"
PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
export PYTHONPATH="$REPO"
mkdir -p logs/slurm

# Avoid NFS stale-file-handle errors from concurrent numba JIT cache writes
export NUMBA_CACHE_DIR="/tmp/numba_cache_${SLURM_JOB_ID}"
mkdir -p "$NUMBA_CACHE_DIR"

echo "======================================================================"
echo "Job:        $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Start:      $(date)"
echo "Result dir: $RESULT_DIR"
echo "======================================================================"

$PYTHON scripts/run_contrastive_eval.py "$RESULT_DIR"

echo ""
echo "End: $(date)"
