#!/usr/bin/env bash
# submit_3ch_pza_jobs.sh
# Train 2 ConAE 3ch paz+zyx+act models on vinc (ds1) only.
# Eval (UMAP+clusters) and cross-dataset eval on all 4 datasets submitted as dependencies.

set -eo pipefail
mkdir -p logs/slurm

PYTHON=/home/liyading/miniconda3/bin/python3
REPO=/net/projects/CLS/lding/gitcode/SubCellAE
RUNNER=$REPO/scripts/run_ae_pipeline.py
EVAL=$REPO/scripts/run_contrastive_eval.py
XEVAL=$REPO/scripts/run_cross_dataset_eval.py
CONFIG_DIR=$REPO/config/contrastive_config
ROOT=/net/projects/CLS/lding/data/fa_data_analysis
RUNS=$ROOT/ae_results/contrastive_run

TRAIN_ARGS="--partition=general --gres=gpu:a40:1 --cpus-per-task=8 --mem=32G --time=08:00:00"
EVAL_ARGS="--partition=general --gres=gpu:a40:1 --cpus-per-task=8 --mem=48G --time=04:00:00"
XEVAL_ARGS="--partition=general --gres=gpu:a40:1 --cpus-per-task=8 --mem=48G --time=04:00:00"

export PYTHONPATH="$REPO:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages"

submit_chain() {
    local cfg_name=$1
    local run_name=$2
    local RUN_DIR="$RUNS/$run_name"

    TRAIN_JOB=$(sbatch --parsable $TRAIN_ARGS \
        --job-name="train_${run_name:0:30}" \
        --output="logs/slurm/${run_name}_train_%j.out" \
        --wrap="exec 2>&1
export PYTHONPATH='$PYTHONPATH'
echo Node: \$(hostname); echo Start: \$(date)
$PYTHON $RUNNER $CONFIG_DIR/${cfg_name}.yaml
echo End: \$(date)")
    echo "  TRAIN  $run_name -> job $TRAIN_JOB"

    EVAL_JOB=$(sbatch --parsable $EVAL_ARGS \
        --dependency=afterok:$TRAIN_JOB \
        --job-name="eval_${run_name:0:30}" \
        --output="logs/slurm/${run_name}_eval_%j.out" \
        --wrap="exec 2>&1
export PYTHONPATH='$PYTHONPATH'
echo Node: \$(hostname); echo Start: \$(date)
$PYTHON $EVAL $RUN_DIR
echo End: \$(date)")
    echo "  EVAL   $run_name -> job $EVAL_JOB (after $TRAIN_JOB)"

    XEVAL_JOB=$(sbatch --parsable $XEVAL_ARGS \
        --dependency=afterok:$TRAIN_JOB \
        --job-name="xeval_${run_name:0:30}" \
        --output="logs/slurm/${run_name}_xeval_%j.out" \
        --wrap="exec 2>&1
export PYTHONPATH='$PYTHONPATH'
echo Node: \$(hostname); echo Start: \$(date)
$PYTHON $XEVAL $RUN_DIR --mode sweep --root-folder $ROOT
echo End: \$(date)")
    echo "  XEVAL  $run_name -> job $XEVAL_JOB (after $TRAIN_JOB)"
    echo ""
}

echo "=== Submitting 3ch pza (pax+zyx+act) jobs ==="
submit_chain \
    "ae_contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_3ch_pza" \
    "contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_3ch_pza"

submit_chain \
    "ae_contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_lc025_3ch_pza" \
    "contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_lc025_3ch_pza"

echo "All jobs submitted. 2 train + 2 eval + 2 xeval = 6 total jobs."
