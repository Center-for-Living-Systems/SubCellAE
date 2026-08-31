#!/usr/bin/env bash
#SBATCH --job-name=ilastik_feat
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/slurm/ilastik_feat_%j.out

set -eo pipefail
exec 2>&1

REPO="$PWD"
PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
export PYTHONPATH="$REPO"
mkdir -p logs/slurm

echo "Node: $(hostname)"
echo "Start: $(date)"
time $PYTHON scripts/extract_ilastik_features.py --all
echo "End: $(date)"
