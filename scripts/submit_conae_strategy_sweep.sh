#!/usr/bin/env bash
# Submit the conAE strategy sweep pipeline to SLURM.
#
# Stage 1: 11 training jobs run in parallel (array 0-10, 1 GPU each)
# Stage 2-4: analysis+cls+vis starts only after all training jobs complete
#
# Usage (from repo root):
#   bash scripts/submit_conae_strategy_sweep.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$REPO_DIR/logs/slurm"

TRAIN_JID=$(sbatch --parsable --chdir="$REPO_DIR" "$REPO_DIR/scripts/sbatch_train_conae_strategy_sweep.sh")
echo "Submitted conAE strategy training array: $TRAIN_JID (strategies: 0322..warmup100_nowd)"

ANALYSIS_JID=$(sbatch --parsable --chdir="$REPO_DIR" \
  --dependency=afterok:${TRAIN_JID} \
  "$REPO_DIR/scripts/sbatch_analysis_conae_strategy_sweep.sh")
echo "Submitted conAE analysis+cls+vis job:    $ANALYSIS_JID (starts after all training complete)"

echo ""
echo "Monitor with:  squeue -u \$USER"
echo "Training logs: $REPO_DIR/logs/slurm/conae_strategy_train_${TRAIN_JID}_[0-10].out"
echo "Analysis log:  $REPO_DIR/logs/slurm/conae_strategy_analysis_${ANALYSIS_JID}.out"
