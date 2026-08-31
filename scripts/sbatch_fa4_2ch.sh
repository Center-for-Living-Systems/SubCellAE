#!/bin/bash
#SBATCH --job-name=fa4_2ch
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/fa4_2ch_%j.log

REPO=/net/projects/CLS/lding/gitcode/SubCellAE
PYTHON=/net/projects/CLS/lding/conda_env/core_env/bin/python
cd "$REPO"
mkdir -p logs
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "FA4 cross-dataset — Option C (stage2_2ch_s3v1, pax+actin)"
$PYTHON scripts/run_fa4_xds_cls.py --mode all --option C --device auto
