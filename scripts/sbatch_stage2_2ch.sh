#!/bin/bash
#SBATCH --job-name=stage2_2ch
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=logs/stage2_2ch_%j.log

REPO=/net/projects/CLS/lding/gitcode/SubCellAE
PYTHON=/net/projects/CLS/lding/conda_env/core_env/bin/python
cd "$REPO"
mkdir -p logs
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "Stage-2 2ch (pax+actin) SupCon AE — all splits"
$PYTHON scripts/run_stage2_2ch_ae_training.py --all-splits --epochs 300
