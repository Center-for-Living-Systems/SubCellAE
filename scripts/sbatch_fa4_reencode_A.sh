#!/bin/bash
#SBATCH --job-name=fa4_reenc_A
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/fa4_reencode_A_%j.log

# Re-encode Option A with both z_recon + z_proj saved (needed for zproj/both variants)
REPO=/net/projects/CLS/lding/gitcode/SubCellAE
PYTHON=/net/projects/CLS/lding/conda_env/core_env/bin/python
cd "$REPO"
mkdir -p logs
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

$PYTHON scripts/run_fa4_xds_cls.py --mode encode --option A --device auto
