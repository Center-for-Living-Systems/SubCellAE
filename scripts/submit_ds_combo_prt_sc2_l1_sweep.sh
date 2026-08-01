#!/usr/bin/env bash
# submit_ds_combo_prt_sc2_l1_sweep.sh
# cio_mode_prt / ÷2 / no clip / no sigmoid / L1
#
# Usage:
#   bash scripts/submit_ds_combo_prt_sc2_l1_sweep.sh            # train + eval
#   bash scripts/submit_ds_combo_prt_sc2_l1_sweep.sh --eval-only

set -eo pipefail

mkdir -p logs/slurm

COMBO_LIST="config/contrastive_config/ds_combo_v4/combo_list.txt"
N=$(wc -l < "$COMBO_LIST")
echo "Submitting ds_combo prt_sc2 L1 sweep  ($N combos)"
cat "$COMBO_LIST"
echo ""

EVAL_ONLY=false
for arg in "$@"; do
    [[ "$arg" == "--eval-only" ]] && EVAL_ONLY=true
done

if [ "$EVAL_ONLY" = false ]; then
    TRAIN_JOB=$(sbatch --parsable \
        --array=0-$((N-1)) \
        scripts/sbatch_train_ds_combo_prt_sc2_l1_sweep.sh)
    echo "Training array job : $TRAIN_JOB  (tasks 0–$((N-1)))"
    DEPEND="--dependency=afterok:${TRAIN_JOB}"
else
    DEPEND=""
fi

EVAL_JOB=$(sbatch --parsable $DEPEND scripts/sbatch_eval_ds_combo_prt_sc2_l1_sweep.sh)
echo "Eval job           : $EVAL_JOB  ${DEPEND}"
