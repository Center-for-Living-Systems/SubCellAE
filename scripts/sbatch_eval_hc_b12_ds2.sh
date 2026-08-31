#!/bin/bash
#SBATCH --job-name=hc_b12_ds2
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/eval_hc_b12_ds2_%j.log

REPO=/net/projects/CLS/lding/gitcode/SubCellAE
PYTHON=/net/projects/CLS/lding/conda_env/core_env/bin/python
cd "$REPO"
mkdir -p logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

echo "=== CellProfiler B12 DS2 lgbm ==="
$PYTHON scripts/eval_handcrafted_features.py --feature cp --label-set b12 --classifier lgbm --dataset ds2

echo "=== CellProfiler B12 DS2 logreg ==="
$PYTHON scripts/eval_handcrafted_features.py --feature cp --label-set b12 --classifier logreg --dataset ds2

echo "=== ilastik B12 DS2 lgbm ==="
$PYTHON scripts/eval_handcrafted_features.py --feature ilastik --label-set b12 --classifier lgbm --dataset ds2

echo "=== ilastik B12 DS2 logreg ==="
$PYTHON scripts/eval_handcrafted_features.py --feature ilastik --label-set b12 --classifier logreg --dataset ds2

echo "Done."
