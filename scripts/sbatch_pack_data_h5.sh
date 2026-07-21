#!/usr/bin/env bash
#SBATCH --job-name=pack_data_h5
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/pack_data_h5_%j.out

set -eo pipefail
exec 2>&1

REPO="$PWD"
PYTHON=/home/liyading/miniconda3/bin/python3
export PYTHONPATH="$REPO:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages"

mkdir -p logs/slurm

echo "[$(date)] Node: $(hostname)"
$PYTHON scripts/pack_data_h5.py "$@"
echo "[$(date)] Done — data.h5 for all 4 datasets"
