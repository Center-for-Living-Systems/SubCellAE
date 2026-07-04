#!/usr/bin/env bash
# Submit the conAE dim sweep pipeline to SLURM.
#
# Stage 1: 7 training jobs in parallel (array job, 1 GPU each)
# Stage 2-4: analysis + cls + vis after all training completes
#
# Usage (from repo root):
#   bash scripts/submit_conae_sweep.sh

set -eo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$REPO_DIR/logs/slurm"

TRAIN_JID=$(sbatch --parsable --chdir="$REPO_DIR" "$REPO_DIR/scripts/sbatch_train_conae_sweep.sh")
echo "Submitted conAE training array job: $TRAIN_JID  (combos: lat12proj12, lat16proj8, lat16proj12, lat24proj8, lat24proj12, lat32proj8, lat32proj12)"

ANA_JID=$(sbatch --parsable --chdir="$REPO_DIR" \
  --dependency=afterok:${TRAIN_JID} \
  "$REPO_DIR/scripts/sbatch_analysis_conae_sweep.sh")
echo "Submitted analysis+cls+vis job:     $ANA_JID (starts after all training complete)"

echo ""
echo "Monitor with:  squeue -u \$USER"
echo "Training logs: $REPO_DIR/logs/slurm/conae_train_${TRAIN_JID}_[0-6].out"
echo "Analysis log:  $REPO_DIR/logs/slurm/conae_analysis_${ANA_JID}.out"
