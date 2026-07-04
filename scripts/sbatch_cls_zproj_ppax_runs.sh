#!/usr/bin/env bash
#SBATCH --job-name=cls_zproj_ppax
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/slurm/cls_zproj_ppax_%j.out

# Classification using projector (p_, 8-dim) features for all 4 vinc+ppax runs
# FA type + position × LightGBM + MLP

set -eo pipefail
exec 2>&1

REPO="$PWD"
PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
export PYTHONPATH="$REPO"
mkdir -p logs/slurm

echo "======================================================================"
echo "Job:   $SLURM_JOB_ID"
echo "Node:  $(hostname)"
echo "Start: $(date)"
echo "======================================================================"

CFG="config/contrastive_config"

RUNS=(
    "supcon_cio_rb_vinc_ppax_lat12proj8_enlcrop_sc2_l1"
    "supcon_cio_rb_vinc_ppax_lat12proj8_enlcrop_sc2_nl1_lc025"
    "contrastive_cio_rb_vinc_ppax_lat12proj8_enlcrop_sc2_nl1"
    "contrastive_cio_rb_vinc_ppax_lat12proj8_enlcrop_sc2_nl1_lc025"
)

for run in "${RUNS[@]}"; do
    for target in fa pos; do
        for clf in lgbm mlp; do
            echo "--- $run | $target | zproj | $clf ---"
            $PYTHON scripts/run_classification_from_config.py \
                $CFG/cls_${run}_${target}_zproj_${clf}.yaml
        done
    done
done

echo ""
echo "End: $(date)"
