#!/usr/bin/env bash
#SBATCH --job-name=cluster_panels
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/cluster_panels_%j.out

# KMeans k=10 + 16-patch center panels for the 3 actin-only (ch3) models.

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

for MODEL_DIR in \
    "$RUNS/baseline_vinc_only_ch3" \
    "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_ch3" \
    "$RUNS/contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_lc025_ch3"
do
    echo ""
    echo "--- $(basename $MODEL_DIR) ---"
    $PYTHON scripts/run_cluster_panels.py "$MODEL_DIR" --k 10 --n-panel 16
done

echo ""
echo "End: $(date)"
