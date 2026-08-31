#!/bin/bash
#SBATCH --job-name=fa4_var_A
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/fa4_variants_A_%j.log

# Run classify + plot for zproj and smote variants (Option A)
# PREREQUISITE: sbatch_fa4_reencode_A.sh must have completed first
REPO=/net/projects/CLS/lding/gitcode/SubCellAE
PYTHON=/net/projects/CLS/lding/conda_env/core_env/bin/python
cd "$REPO"
mkdir -p logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

echo "=== zproj only ==="
$PYTHON scripts/run_fa4_xds_cls.py --mode classify --option A --variant zproj
$PYTHON scripts/run_fa4_xds_cls.py --mode plot     --option A --variant zproj
$PYTHON scripts/run_fa4_xds_cls.py --mode ppax_zeroshot --option A --variant zproj --device cpu

echo "=== zrecon + SMOTE ==="
$PYTHON scripts/run_fa4_xds_cls.py --mode classify --option A --variant zrecon --smote
$PYTHON scripts/run_fa4_xds_cls.py --mode plot     --option A --variant zrecon --smote
$PYTHON scripts/run_fa4_xds_cls.py --mode ppax_zeroshot --option A --variant zrecon --smote --device cpu

echo "=== zproj + SMOTE ==="
$PYTHON scripts/run_fa4_xds_cls.py --mode classify --option A --variant zproj --smote
$PYTHON scripts/run_fa4_xds_cls.py --mode plot     --option A --variant zproj --smote
$PYTHON scripts/run_fa4_xds_cls.py --mode ppax_zeroshot --option A --variant zproj --smote --device cpu
