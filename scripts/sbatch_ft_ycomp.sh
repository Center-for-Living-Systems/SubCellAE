#!/bin/bash
#SBATCH --job-name=ft_ycomp
#SBATCH --array=0-9
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/ft_ycomp_%A_%a.log

# Array index → (base, mode, frac)
# 0-3: corrected_s3v1 full_ft fracs 0.10/0.25/0.50/0.75
# 4-7: combined_s3v1  full_ft fracs 0.10/0.25/0.50/0.75
# 8:   corrected_s3v1 cls_only
# 9:   combined_s3v1  cls_only

BASES=(corrected_s3v1 corrected_s3v1 corrected_s3v1 corrected_s3v1
       combined_s3v1  combined_s3v1  combined_s3v1  combined_s3v1
       corrected_s3v1 combined_s3v1)
MODES=(full_ft full_ft full_ft full_ft
       full_ft full_ft full_ft full_ft
       cls_only cls_only)
FRACS=(0.10 0.25 0.50 0.75
       0.10 0.25 0.50 0.75
       0.10 0.10)   # frac ignored for cls_only (all fracs run at once)

BASE=${BASES[$SLURM_ARRAY_TASK_ID]}
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}
FRAC=${FRACS[$SLURM_ARRAY_TASK_ID]}

REPO=/net/projects/CLS/lding/gitcode/SubCellAE
PYTHON=/net/projects/CLS/lding/conda_env/core_env/bin/python

cd "$REPO"
mkdir -p logs

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

if [ "$MODE" = "cls_only" ]; then
    echo "Running cls_only  base=$BASE"
    $PYTHON scripts/run_finetune_ycomp.py \
        --base "$BASE" \
        --mode cls_only
else
    echo "Running full_ft  base=$BASE  frac=$FRAC"
    $PYTHON scripts/run_finetune_ycomp.py \
        --base "$BASE" \
        --mode full_ft \
        --frac "$FRAC" \
        --epochs 100 \
        --lr 2e-4
fi
