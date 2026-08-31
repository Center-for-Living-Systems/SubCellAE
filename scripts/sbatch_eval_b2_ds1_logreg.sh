#!/bin/bash
#SBATCH --job-name=eval_b2_ds1_lr
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_b2_ds1_logreg_%j.log

REPO=/net/projects/CLS/lding/gitcode/SubCellAE
PYTHON=/net/projects/CLS/lding/conda_env/core_env/bin/python
cd "$REPO"
mkdir -p logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

echo "=== CP B2 DS1 logreg ==="
$PYTHON scripts/eval_handcrafted_features.py --feature cp --label-set b2 --classifier logreg --dataset ds1

echo "=== ilastik B2 DS1 logreg ==="
$PYTHON scripts/eval_handcrafted_features.py --feature ilastik --label-set b2 --classifier logreg --dataset ds1

echo "=== SupCon B2 DS1 logreg ==="
$PYTHON scripts/eval_supcon_latents.py --run-tag le_b2_supcon --classifier logreg --dataset ds1

echo "Done."
