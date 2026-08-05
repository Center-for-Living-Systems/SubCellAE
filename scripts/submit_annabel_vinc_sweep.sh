#!/usr/bin/env bash
# submit_annabel_vinc_sweep.sh
#
# Annabel vinc control full sweep:
#   train (9) → eval (9) + cls (18) in parallel after train
#
# Depends on export job 1326749 (cio_mode_prt source frames).
#
# Usage:
#   bash scripts/submit_annabel_vinc_sweep.sh            # full pipeline
#   bash scripts/submit_annabel_vinc_sweep.sh --eval-only
#   bash scripts/submit_annabel_vinc_sweep.sh --after <jobid>

set -eo pipefail
mkdir -p logs/slurm

EXPORT_JOB=1326749   # cio_mode_prt frame export

EVAL_ONLY=false
AFTER_JOB=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --eval-only) EVAL_ONLY=true ;;
        --after)     AFTER_JOB="$2"; shift ;;
    esac
    shift
done

if [ "$EVAL_ONLY" = false ]; then
    TRAIN_JOB=$(sbatch --parsable \
        --array=0-8 \
        scripts/sbatch_train_annabel_vinc.sh)
    echo "Training  : $TRAIN_JOB  (tasks 0-8)"
    POST_DEP="--dependency=afterok:${TRAIN_JOB}"
else
    if [ -n "$AFTER_JOB" ]; then
        POST_DEP="--dependency=afterok:${AFTER_JOB}"
    else
        POST_DEP=""
    fi
fi

EVAL_JOB=$(sbatch --parsable $POST_DEP \
    --array=0-8 \
    scripts/sbatch_eval_annabel_vinc.sh)
echo "Eval      : $EVAL_JOB  (tasks 0-8)"

CLS_JOB=$(sbatch --parsable $POST_DEP \
    --array=0-17 \
    scripts/sbatch_cls_annabel_vinc.sh)
echo "Cls       : $CLS_JOB  (tasks 0-17)"

PACK_JOB=$(sbatch --parsable \
    --dependency=afterok:${EVAL_JOB}:${CLS_JOB} \
    --array=0-8 \
    scripts/sbatch_pack_annabel_vinc.sh)
echo "Pack      : $PACK_JOB  (tasks 0-8, after eval + cls)"
