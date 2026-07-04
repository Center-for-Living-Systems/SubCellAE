#!/usr/bin/env bash
#SBATCH --job-name=eval_ppax_runs
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm/eval_ppax_runs_%j.out

# Eval pipeline for all 4 vinc+ppax balanced contrastive runs:
#   1. KNN eval (run_contrastive_eval.py)
#   2. LightGBM + MLP classification (FA type + position, z_recon features)

set -eo pipefail
exec 2>&1

REPO="$PWD"
PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
export PYTHONPATH="$REPO"
export NUMBA_CACHE_DIR="/tmp/numba_cache_${SLURM_JOB_ID}"
mkdir -p logs/slurm "$NUMBA_CACHE_DIR"

echo "======================================================================"
echo "Job:   $SLURM_JOB_ID"
echo "Node:  $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Start: $(date)"
echo "======================================================================"

ROOT="/net/projects/CLS/lding/data/fa_data_analysis"
CFG="config/contrastive_config"

RUNS=(
    "supcon_cio_rb_vinc_ppax_lat12proj8_enlcrop_sc2_l1"
    "supcon_cio_rb_vinc_ppax_lat12proj8_enlcrop_sc2_nl1_lc025"
    "contrastive_cio_rb_vinc_ppax_lat12proj8_enlcrop_sc2_nl1"
    "contrastive_cio_rb_vinc_ppax_lat12proj8_enlcrop_sc2_nl1_lc025"
)

echo "======================================================================"
echo "[$(date)] STAGE 1 — KNN eval (vinc val + ppax transfer)"
echo "======================================================================"

for run in "${RUNS[@]}"; do
    echo "--- KNN: $run ---"
    $PYTHON scripts/run_contrastive_eval.py \
        "$ROOT/ae_results/contrastive_run/$run"
done

echo ""
echo "======================================================================"
echo "[$(date)] STAGE 2 — LightGBM + MLP classification"
echo "======================================================================"

for run in "${RUNS[@]}"; do
    for target in fa pos; do
        for clf in lgbm mlp; do
            echo "--- cls: $run | $target | $clf ---"
            $PYTHON scripts/run_classification_from_config.py \
                $CFG/cls_${run}_${target}_zrecon_${clf}.yaml
        done
    done
done

echo ""
echo "======================================================================"
echo "[$(date)] ALL DONE"
echo "======================================================================"
echo "End: $(date)"
