#!/bin/bash
#SBATCH --job-name=eval_sc_b12_ds2
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_supcon_b12_ds2_%j.log

REPO=/net/projects/CLS/lding/gitcode/SubCellAE
PYTHON=/net/projects/CLS/lding/conda_env/core_env/bin/python
cd "$REPO"
mkdir -p logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

echo "=== SupCon B12 DS2 logreg ==="
$PYTHON scripts/eval_supcon_latents.py --run-tag le_b12_supcon --classifier logreg --dataset ds2

echo "Done."
