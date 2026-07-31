#!/usr/bin/env bash
#SBATCH --job-name=panels_ds_combo_v3
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/cluster_panels_ds_combo_v3_%j.out

# KMeans k=10 cluster panels for all 30 ds_combo_v3 models:
#   15 × clip01_l1 (CIO clip[0,1], no sc2, L1)
#   15 × sc2_clip02_l1 (CIO clip[0,2]+sc2÷2, L1)

set -eo pipefail
exec 2>&1

REPO="$PWD"
PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
export PYTHONPATH="$REPO"
mkdir -p logs/slurm

RUNS="/net/projects/CLS/lding/data/fa_data_analysis/ae_results/contrastive_run"
COMBO_LIST="config/contrastive_config/ds_combo_v3/combo_list.txt"

echo "======================================================================"
echo "Job:   $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"
echo "======================================================================"

echo ""
echo "=== ds_combo_enlcrop_clip01_l1 (15 combos) ==="
while IFS= read -r COMBO; do
    MODEL_DIR="$RUNS/ds_combo_enlcrop_clip01_l1/$COMBO"
    if [ ! -f "$MODEL_DIR/latents.csv" ]; then
        echo "  SKIP $COMBO (no latents.csv)"
        continue
    fi
    echo ""
    echo "--- clip01_l1 / $COMBO ---"
    $PYTHON scripts/run_cluster_panels.py "$MODEL_DIR" --k 10 --n-panel 16
done < "$COMBO_LIST"

echo ""
echo "=== ds_combo_enlcrop_sc2_clip02_l1 (15 combos) ==="
while IFS= read -r COMBO; do
    MODEL_DIR="$RUNS/ds_combo_enlcrop_sc2_clip02_l1/$COMBO"
    if [ ! -f "$MODEL_DIR/latents.csv" ]; then
        echo "  SKIP $COMBO (no latents.csv)"
        continue
    fi
    echo ""
    echo "--- sc2_clip02_l1 / $COMBO ---"
    $PYTHON scripts/run_cluster_panels.py "$MODEL_DIR" --k 10 --n-panel 16
done < "$COMBO_LIST"

echo ""
echo "End: $(date)"
