#!/bin/bash
#SBATCH --job-name=eval_fa4_b12
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/eval_fa4_b12_%j.log

REPO=/net/projects/CLS/lding/gitcode/SubCellAE
PYTHON=/net/projects/CLS/lding/conda_env/core_env/bin/python
cd "$REPO"
mkdir -p logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

echo "=== FA4 4-class CV eval (B1-only + B12 lat32p16) ==="
$PYTHON scripts/eval_fa4_features.py

echo "Done."
