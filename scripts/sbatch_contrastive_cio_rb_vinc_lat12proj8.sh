#!/usr/bin/env bash
#SBATCH --job-name=vinc_contrastive
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

set -euo pipefail

REPO="/net/projects/CLS/lding/gitcode/SubCellAE_contrastive_projector"
cd "$REPO"
mkdir -p logs

set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate subcellae-cuda
set -u

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start:  $(date)"
echo ""

bash scripts/run_contrastive_cio_rb_vinc_lat12proj8.sh 2>&1 | tee logs/vinc_run.log

echo ""
echo "End: $(date)"
