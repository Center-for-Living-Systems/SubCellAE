#!/bin/bash
#SBATCH --job-name=fa4_b12
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/stage2_fa4_b12_%a_%j.log
#SBATCH --array=0-2

REPO=/net/projects/CLS/lding/gitcode/SubCellAE
PYTHON=/net/projects/CLS/lding/conda_env/core_env/bin/python
cd "$REPO"
mkdir -p logs
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

VERSION=$SLURM_ARRAY_TASK_ID
echo "FA4 B12 Stage-2 SupCon  lat=32 proj=16  version=${VERSION}"
$PYTHON scripts/run_stage2_fa4_b12.py --version ${VERSION} --epochs 300
