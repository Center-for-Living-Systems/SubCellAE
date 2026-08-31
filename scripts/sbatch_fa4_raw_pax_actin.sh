#!/bin/bash
#SBATCH --job-name=fa4_raw_pax_act
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=logs/fa4_raw_pax_actin_%j.log

REPO=/net/projects/CLS/lding/gitcode/SubCellAE
PYTHON=/net/projects/CLS/lding/conda_env/core_env/bin/python
cd "$REPO"
mkdir -p logs
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "FA4 raw pixel stats — pax + actin channels"
$PYTHON scripts/run_fa4_raw_cls.py --mode all --channels pax_actin
