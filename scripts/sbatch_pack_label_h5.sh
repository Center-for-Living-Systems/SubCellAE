#!/usr/bin/env bash
#SBATCH --job-name=pack_label_h5
#SBATCH --partition=general
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/pack_label_h5_%j.out

set -eo pipefail
exec 2>&1

mkdir -p logs/slurm

echo "======================================================================"
echo "[$(date)] Node: $(hostname)"
echo "======================================================================"

echo "[$(date)] Packing label H5 files (all 4 datasets × 2 conditions)..."
/net/projects/CLS/lding/conda_env/core_env/bin/python3 scripts/pack_patches_label_h5.py

echo "[$(date)] ALL DONE"
