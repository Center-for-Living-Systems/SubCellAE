#!/usr/bin/env bash
#SBATCH --job-name=vinc_cls_vis
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

REPO="/net/projects/CLS/lding/gitcode/SubCellAE_contrastive_projector"
cd "$REPO"
mkdir -p logs

set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate subcellae-cuda
set -u

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "Start:  $(date)"
echo ""

bash scripts/run_contrastive_cio_rb_vinc_lat12proj8_cls_vis.sh 2>&1 | tee logs/vinc_cls_vis.log

echo ""
echo "End: $(date)"
