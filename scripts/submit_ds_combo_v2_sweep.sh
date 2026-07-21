#!/usr/bin/env bash
# submit_ds_combo_v2_sweep.sh
#
# Submit the v2 ds_combo training array + chained eval job.
# v2 changes: lambda_contrast=0.10, ds1 40%/60% split, ds2×3, ds3×2, ds4×2.
#
# Usage:
#   bash scripts/submit_ds_combo_v2_sweep.sh            # train + eval
#   bash scripts/submit_ds_combo_v2_sweep.sh --eval-only # resubmit eval only

set -eo pipefail

mkdir -p logs/slurm

COMBO_LIST="config/contrastive_config/ds_combo_v2/combo_list.txt"
N=$(wc -l < "$COMBO_LIST")
echo "Submitting v2 ds_combo sweep  ($N combos)"
cat "$COMBO_LIST"
echo ""

EVAL_ONLY=false
for arg in "$@"; do
    [[ "$arg" == "--eval-only" ]] && EVAL_ONLY=true
done

if [ "$EVAL_ONLY" = false ]; then
    TRAIN_JOB=$(sbatch --parsable \
        --array=0-$((N-1)) \
        scripts/sbatch_train_ds_combo_v2_sweep.sh)
    echo "Training array job : $TRAIN_JOB  (tasks 0–$((N-1)))"
    DEPEND="--dependency=afterok:${TRAIN_JOB}"
else
    DEPEND=""
    echo "Skipping training (--eval-only)"
fi

EVAL_JOB=$(sbatch --parsable $DEPEND \
    scripts/sbatch_eval_ds_combo_v2_sweep.sh)
echo "Eval+analysis job  : $EVAL_JOB  ${DEPEND:+(afterok $TRAIN_JOB)}"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f logs/slurm/ds_combo_v2_train_${TRAIN_JOB:-?}_0.out"
