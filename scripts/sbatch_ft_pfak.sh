#!/bin/bash
#SBATCH --job-name=ft_pfak
#SBATCH --array=0-3
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/ft_pfak_%A_%a.log

# Array index → frac
FRACS=(0.10 0.25 0.50 0.75)
FRAC=${FRACS[$SLURM_ARRAY_TASK_ID]}

REPO=/net/projects/CLS/lding/gitcode/SubCellAE
PYTHON=/net/projects/CLS/lding/conda_env/core_env/bin/python

cd "$REPO"
mkdir -p logs

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "Running pfak full_ft  frac=$FRAC"
$PYTHON scripts/run_finetune_pfak.py \
    --mode full_ft \
    --frac "$FRAC" \
    --epochs 100 \
    --lr 2e-4
