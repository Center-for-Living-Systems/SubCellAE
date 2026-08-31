#!/bin/bash
#SBATCH --job-name=eval_hc_b12
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_hc_b12_%j.log

# Evaluate CellProfiler and ilastik features on B1+B2 combined DS1 splits
# label efficiency setting (5-fold CV, all budgets)

REPO=/net/projects/CLS/lding/gitcode/SubCellAE
PYTHON=/net/projects/CLS/lding/conda_env/core_env/bin/python
cd "$REPO"
mkdir -p logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

echo "=== CellProfiler B12 DS1 ==="
$PYTHON scripts/eval_handcrafted_features.py --feature cp --label-set b12 --dataset ds1

echo "=== ilastik B12 DS1 ==="
$PYTHON scripts/eval_handcrafted_features.py --feature ilastik --label-set b12 --dataset ds1

echo "Done."
