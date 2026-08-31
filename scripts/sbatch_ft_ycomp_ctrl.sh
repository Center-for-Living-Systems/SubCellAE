#!/bin/bash
#SBATCH --job-name=ft_ycomp_ctrl
#SBATCH --array=0-4
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=logs/ft_ycomp_ctrl_%A_%a.log

# Array: 0=frac0.00, 1=frac0.10, 2=frac0.25, 3=frac0.50, 4=frac0.75
FRACS=(0.00 0.10 0.25 0.50 0.75)
FRAC=${FRACS[$SLURM_ARRAY_TASK_ID]}

REPO=/net/projects/CLS/lding/gitcode/SubCellAE
PYTHON=/net/projects/CLS/lding/conda_env/core_env/bin/python
cd "$REPO"
mkdir -p logs
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "Running ycomp full_ft --add-ctrl  frac=$FRAC  base=combined_s3v1"
$PYTHON scripts/run_finetune_ycomp.py \
    --base combined_s3v1 \
    --mode full_ft \
    --add-ctrl \
    --frac "$FRAC" \
    --epochs 100 \
    --lr 2e-4
