#!/bin/bash
#SBATCH --job-name=fa4_optA
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/fa4_optionA_%j.log

# Option A: encode new patches with stage2_s3v1, run all experiments + ppax zero-shot
REPO=/net/projects/CLS/lding/gitcode/SubCellAE
PYTHON=/net/projects/CLS/lding/conda_env/core_env/bin/python
cd "$REPO"
mkdir -p logs
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "FA4 cross-dataset — Option A (stage2_s3v1 model)"
$PYTHON scripts/run_fa4_xds_cls.py --mode all --option A --device auto
