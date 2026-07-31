#!/usr/bin/env bash
#SBATCH --job-name=crossds_extra_dsc
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/crossds_extra_dsc_%j.out

# Compute cross-dataset latents + z_proj UMAP/PHATE for the 4 ds_combo sweeps
# not covered by the first crossds_latents job (1224526):
#   ds_combo_enlcrop_sc2
#   ds_combo_enlcrop_sc2_lc010_bal
#   ds_combo_enlcrop_sc2_lc010_bal_l1
#   ds_combo_enlcrop_sc2_lc010_bal_mse

set -eo pipefail
exec 2>&1

REPO="$PWD"
PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
export PYTHONPATH="$REPO"
mkdir -p logs/slurm

RUNS="/net/projects/CLS/lding/data/fa_data_analysis/ae_results/contrastive_run"
COMBO_LIST="config/contrastive_config/ds_combo_v3/combo_list.txt"
CROSSDS="$PYTHON scripts/run_crossds_latents.py"

echo "======================================================================"
echo "Job:   $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"
echo "======================================================================"

for PARENT_DIR in \
    "$RUNS/ds_combo_enlcrop_sc2" \
    "$RUNS/ds_combo_enlcrop_sc2_lc010_bal" \
    "$RUNS/ds_combo_enlcrop_sc2_lc010_bal_l1" \
    "$RUNS/ds_combo_enlcrop_sc2_lc010_bal_mse"
do
    PARENT_NAME=$(basename "$PARENT_DIR")
    echo ""
    echo "=== $PARENT_NAME ==="
    while IFS= read -r COMBO; do
        MODEL_DIR="$PARENT_DIR/$COMBO"
        [ -f "$MODEL_DIR/latents.csv" ] || { echo "  SKIP $COMBO (no latents.csv)"; continue; }
        echo ""
        echo "--- $PARENT_NAME / $COMBO ---"
        $CROSSDS "$MODEL_DIR" || echo "  WARNING: failed for $COMBO"
    done < "$COMBO_LIST"
done

echo ""
echo "End: $(date)"
