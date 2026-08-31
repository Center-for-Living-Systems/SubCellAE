#!/bin/bash
#SBATCH --job-name=fa4_optB
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/fa4_optionB_%j.log

# Option B: encode with stage2_combined model, run all experiments + ppax zero-shot
# PREREQUISITE: sbatch_stage2_combined.sh must have completed first
REPO=/net/projects/CLS/lding/gitcode/SubCellAE
PYTHON=/net/projects/CLS/lding/conda_env/core_env/bin/python
cd "$REPO"
mkdir -p logs
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "FA4 cross-dataset — Option B (stage2_combined model)"
$PYTHON scripts/run_fa4_xds_cls.py --mode all --option B --device auto
