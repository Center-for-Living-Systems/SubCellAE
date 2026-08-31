#!/bin/bash
#SBATCH --job-name=eval_sc_vc
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_supcon_vc_%j.log

REPO=/net/projects/CLS/lding/gitcode/SubCellAE
PYTHON=/net/projects/CLS/lding/conda_env/core_env/bin/python
cd "$REPO"
mkdir -p logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

echo "=== SupCon vinc-ctrl-only (le_b2_vinc_ctrl) logreg ==="
$PYTHON scripts/eval_supcon_latents.py --run-tag le_b2_vinc_ctrl --classifier logreg --dataset vc

echo "Done."
