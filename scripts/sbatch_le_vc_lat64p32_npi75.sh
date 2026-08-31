#!/usr/bin/env bash
#SBATCH --job-name=le_vc_64p32
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm/le_vc_lat64p32_%j.out

set -eo pipefail
exec 2>&1

REPO="$PWD"
PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
export PYTHONPATH="$REPO"
mkdir -p logs/slurm

NAME="le_vc_lat64p32_fv0_nb75_r0"
CONFIG="config/le_vc_lat64p32/${NAME}.yaml"

DATA="/net/projects/CLS/lding/data/fa_data_analysis"
ANN_CSV="${DATA}/labelling/le_b2_vinc_ctrl/le_b2_vinc_ctrl_fv0_nb75_r0.csv"
FOLD_SPLITS="${DATA}/labelling/le_b2_vinc_ctrl/fold_splits.csv"
RUN_DIR="${DATA}/ae_results/contrastive_run/le_vc_lat64p32/${NAME}"

echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Start: $(date)"

$PYTHON scripts/run_ae_from_config.py "$CONFIG"

echo "--- LGBM eval ---"
$PYTHON scripts/eval_one_supcon_run.py \
    --run-dir     "$RUN_DIR" \
    --ann-csv     "$ANN_CSV" \
    --fold-splits "$FOLD_SPLITS" \
    --fold        0 \
    --budget      75 \
    --repeat      0

echo "End: $(date)"
