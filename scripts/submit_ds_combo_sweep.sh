#!/usr/bin/env bash
# submit_ds_combo_sweep.sh
#
# Full orchestration for the dataset-combination ConAE sweep:
#   1. Generate 15 training configs (if not already done)
#   2. Submit training array job (15 tasks, GPU)
#   3. Chain eval+analysis job (afterok all 15 tasks)
#
# Usage:
#   bash scripts/submit_ds_combo_sweep.sh            # submit everything
#   bash scripts/submit_ds_combo_sweep.sh --eval-only # resubmit eval only

set -eo pipefail

REPO="$PWD"
export PYTHONPATH="$REPO:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages"
PYTHON=/home/liyading/miniconda3/bin/python3
COMBO_LIST="config/contrastive_config/ds_combo/combo_list.txt"

mkdir -p logs/slurm

EVAL_ONLY=false
for arg in "$@"; do
    [[ "$arg" == "--eval-only" ]] && EVAL_ONLY=true
done

# ── Step 1: generate configs ──────────────────────────────────────────────────
echo "Generating dataset-combo configs…"
$PYTHON scripts/generate_ds_combo_configs.py
echo ""

N=$(wc -l < "$COMBO_LIST")
echo "Combos: $N"
cat "$COMBO_LIST"
echo ""

# ── Step 2: submit training array ─────────────────────────────────────────────
if [ "$EVAL_ONLY" = false ]; then
    TRAIN_JOB=$(sbatch --parsable \
        --array=0-$((N-1)) \
        scripts/sbatch_train_ds_combo_sweep.sh)
    echo "Training array job: $TRAIN_JOB  (tasks 0–$((N-1)))"

    DEPEND="--dependency=afterok:${TRAIN_JOB}"
else
    DEPEND=""
    echo "Skipping training (--eval-only)"
fi

# ── Step 3: submit eval+analysis (chains after all training tasks) ─────────────
EVAL_JOB=$(sbatch --parsable $DEPEND \
    scripts/sbatch_eval_ds_combo_sweep.sh)
echo "Eval+analysis job : $EVAL_JOB  ${DEPEND:+(afterok $TRAIN_JOB)}"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f logs/slurm/ds_combo_train_${TRAIN_JOB:-?}_0.out"
