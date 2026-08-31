#!/bin/bash
#SBATCH --job-name=s2_combined
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/stage2_combined_%j.log

REPO=/net/projects/CLS/lding/gitcode/SubCellAE
PYTHON=/net/projects/CLS/lding/conda_env/core_env/bin/python
cd "$REPO"
mkdir -p logs
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "Training Stage-2 combined AE (Option B)"
$PYTHON scripts/run_stage2_combined_ae.py --epochs 300 --lr 5e-4 --batch-size 128
