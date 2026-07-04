#!/usr/bin/env bash
#SBATCH --job-name=regen_canvases
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/regen_canvases_%j.out

set -eo pipefail
exec 2>&1

REPO="$PWD"
PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
export PYTHONPATH="$REPO"
mkdir -p logs/slurm

RUNS="/net/projects/CLS/lding/data/fa_data_analysis/ae_results/contrastive_run"

echo "======================================================================"
echo "Job:   $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"
echo "======================================================================"

$PYTHON scripts/regen_canvases.py \
    "$RUNS/contrastive_cio_rb_vinc_ppax_lat12proj8_enlcrop_sc2_nl1" \
    "$RUNS/contrastive_cio_rb_vinc_ppax_lat12proj8_enlcrop_sc2_nl1_lc025" \
    "$RUNS/supcon_cio_rb_vinc_ppax_lat12proj8_enlcrop_sc2_l1" \
    "$RUNS/supcon_cio_rb_vinc_ppax_lat12proj8_enlcrop_sc2_nl1_lc025"

echo ""
echo "End: $(date)"
