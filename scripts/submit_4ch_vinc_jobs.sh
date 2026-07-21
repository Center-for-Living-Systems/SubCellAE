#!/usr/bin/env bash
# submit_4ch_vinc_jobs.sh
# Train 2 ConAE 4ch (vinc+pax+zyx+act) models on vinc (ds1).
# Eval chain: run_contrastive_eval + run_cluster_panels --k 10 + run_cross_dataset_eval
# All eval steps depend on training completing (afterok).

set -eo pipefail
mkdir -p logs/slurm

PYTHON=/home/liyading/miniconda3/bin/python3
REPO=/net/projects/CLS/lding/gitcode/SubCellAE
RUNNER=$REPO/scripts/run_ae_pipeline.py
EVAL=$REPO/scripts/run_contrastive_eval.py
PANELS=$REPO/scripts/run_cluster_panels.py
XEVAL=$REPO/scripts/run_cross_dataset_eval.py
CONFIG_DIR=$REPO/config/contrastive_config
ROOT=/net/projects/CLS/lding/data/fa_data_analysis
RUNS=$ROOT/ae_results/contrastive_run

TRAIN_ARGS="--partition=general --gres=gpu:a40:1 --cpus-per-task=8 --mem=32G --time=08:00:00"
EVAL_ARGS="--partition=general --gres=gpu:a40:1 --cpus-per-task=8 --mem=48G --time=04:00:00"

export PYTHONPATH="$REPO:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages"

submit_chain() {
    local cfg_name=$1
    local run_name=$2
    local RUN_DIR="$RUNS/$run_name"

    # Training
    TRAIN_JOB=$(sbatch --parsable $TRAIN_ARGS \
        --job-name="train_${run_name:0:30}" \
        --output="logs/slurm/${run_name}_train_%j.out" \
        --wrap="exec 2>&1
export PYTHONPATH='$PYTHONPATH'
echo Node: \$(hostname); echo Start: \$(date)
$PYTHON $RUNNER $CONFIG_DIR/${cfg_name}.yaml
echo End: \$(date)")
    echo "  TRAIN   $run_name -> job $TRAIN_JOB"

    # UMAP + annotation eval
    sbatch --parsable $EVAL_ARGS \
        --dependency=afterok:$TRAIN_JOB \
        --job-name="eval_${run_name:0:28}" \
        --output="logs/slurm/${run_name}_eval_%j.out" \
        --wrap="exec 2>&1
export PYTHONPATH='$PYTHONPATH'
echo Node: \$(hostname); echo Start: \$(date)
$PYTHON $EVAL $RUN_DIR --kmeans_k 10
echo End: \$(date)" > /dev/null
    echo "  EVAL    (after $TRAIN_JOB)"

    # KMeans k=10 cluster panels
    sbatch --parsable $EVAL_ARGS \
        --dependency=afterok:$TRAIN_JOB \
        --job-name="panels_${run_name:0:26}" \
        --output="logs/slurm/${run_name}_panels_%j.out" \
        --wrap="exec 2>&1
export PYTHONPATH='$PYTHONPATH'
echo Node: \$(hostname); echo Start: \$(date)
$PYTHON $PANELS $RUN_DIR --k 10
echo End: \$(date)" > /dev/null
    echo "  PANELS  k=10 (after $TRAIN_JOB)"

    # Cross-dataset eval
    sbatch --parsable $EVAL_ARGS \
        --dependency=afterok:$TRAIN_JOB \
        --job-name="xeval_${run_name:0:27}" \
        --output="logs/slurm/${run_name}_xeval_%j.out" \
        --wrap="exec 2>&1
export PYTHONPATH='$PYTHONPATH'
echo Node: \$(hostname); echo Start: \$(date)
$PYTHON $XEVAL $RUN_DIR --mode sweep --root-folder $ROOT
echo End: \$(date)" > /dev/null
    echo "  XEVAL   (after $TRAIN_JOB)"
    echo ""
}

echo "=== Submitting 4ch vinc (vinc+pax+zyx+act) jobs ==="
submit_chain \
    "ae_contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_4ch_vinc" \
    "contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_4ch_vinc"

submit_chain \
    "ae_contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_lc025_4ch_vinc" \
    "contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_lc025_4ch_vinc"

echo "All jobs submitted. 2 train + 6 eval/panels/xeval = 8 total jobs."
