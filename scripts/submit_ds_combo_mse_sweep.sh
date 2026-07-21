#!/usr/bin/env bash
# submit_ds_combo_mse_sweep.sh — MSE loss variant of the v2 balanced sweep.
#
# Usage:
#   bash scripts/submit_ds_combo_mse_sweep.sh            # train + eval
#   bash scripts/submit_ds_combo_mse_sweep.sh --eval-only

set -eo pipefail

mkdir -p logs/slurm

COMBO_LIST="config/contrastive_config/ds_combo_v2/combo_list.txt"
N=$(wc -l < "$COMBO_LIST")
echo "Submitting ds_combo MSE sweep  ($N combos)"
cat "$COMBO_LIST"
echo ""

EVAL_ONLY=false
for arg in "$@"; do
    [[ "$arg" == "--eval-only" ]] && EVAL_ONLY=true
done

if [ "$EVAL_ONLY" = false ]; then
    TRAIN_JOB=$(sbatch --parsable \
        --array=0-$((N-1)) \
        scripts/sbatch_train_ds_combo_mse_sweep.sh)
    echo "Training array job : $TRAIN_JOB  (tasks 0–$((N-1)))"
    DEPEND="--dependency=afterok:${TRAIN_JOB}"
else
    DEPEND=""
    echo "Skipping training (--eval-only)"
fi

EVAL_JOB=$(sbatch --parsable $DEPEND \
    scripts/sbatch_eval_ds_combo_mse_sweep.sh)
echo "Eval+analysis job  : $EVAL_JOB  ${DEPEND:+(afterok $TRAIN_JOB)}"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
