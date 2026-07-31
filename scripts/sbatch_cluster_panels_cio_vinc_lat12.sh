#!/usr/bin/env bash
#SBATCH --job-name=panels_cio_vinc_lat12
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/cluster_panels_cio_vinc_lat12_%j.out

# KMeans k=10 cluster panels for 6 contrastive/supcon cio_vinc lat12proj8 models.
# These have patches_raw.tif so can run immediately.

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
    "$RUNS/contrastive_cio_vinc_lat12proj8_enlcrop_sc2_nl1" \
    "$RUNS/contrastive_cio_vinc_lat12proj8_enlcrop_sc2_nl1_lc100" \
    "$RUNS/contrastive_cio_vinc_lat12proj8_enlcrop_sc2_nl1_lc1e4" \
    "$RUNS/supcon_cio_vinc_lat12proj8_enlcrop_sc2_nl1" \
    "$RUNS/supcon_cio_vinc_lat12proj8_enlcrop_sc2_nl1_lc100" \
    "$RUNS/supcon_cio_vinc_lat12proj8_enlcrop_sc2_nl1_lc1e4"
do
    echo ""
    echo "--- $(basename $MODEL_DIR) ---"
    $PYTHON scripts/run_cluster_panels.py "$MODEL_DIR" --k 10 --n-panel 16
done

echo ""
echo "End: $(date)"
