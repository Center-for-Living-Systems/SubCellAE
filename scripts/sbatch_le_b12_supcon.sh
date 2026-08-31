#!/usr/bin/env bash
#SBATCH --job-name=le_b12_sc
# Submit DS1+DS2 (460 jobs in batches of 200):
#   sbatch --array=0-199   scripts/sbatch_le_b2_supcon.sh
#   sbatch --array=200-399 scripts/sbatch_le_b2_supcon.sh
#   sbatch --array=400-459 scripts/sbatch_le_b2_supcon.sh
# Submit DS3 (180 jobs, use LE_JOB_LIST env var):
#   LE_JOB_LIST=config/le_b2_supcon/job_list_ds3.txt sbatch --array=0-179 scripts/sbatch_le_b2_supcon.sh
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm/le_b12_sc_%A_%a.out

set -eo pipefail
exec 2>&1

REPO="$PWD"
PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
export PYTHONPATH="$REPO"
mkdir -p logs/slurm

JOB_LIST="${LE_JOB_LIST:-config/le_b12_supcon/job_list.txt}"
CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$JOB_LIST")

echo "task $SLURM_ARRAY_TASK_ID -> $CONFIG"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Start: $(date)"

$PYTHON scripts/run_ae_from_config.py "$CONFIG"

# Clean per-epoch checkpoints to save disk space
RESULT_DIR=$($PYTHON -c "
import sys; sys.path.insert(0,'$REPO')
from subcellae.utils.config_utils import resolve_root
import yaml
raw = yaml.safe_load(open('$CONFIG'))
raw = resolve_root(raw)
print(raw['output']['result_dir'])
" 2>/dev/null)
if [ -n "$RESULT_DIR" ] && [ -d "$RESULT_DIR" ]; then
    find "$RESULT_DIR" -name "supcon_model_ep*.pt" -delete 2>/dev/null || true
    echo "Cleaned epoch checkpoints from $RESULT_DIR"
fi

echo "End: $(date)"
