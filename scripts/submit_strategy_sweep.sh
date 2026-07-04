#!/usr/bin/env bash
# Submit the full strategy sweep pipeline to SLURM.
#
# Stage 1: 4 training jobs run in parallel (array job, 1 GPU each)
# Stage 2+3: cls+vis job starts only after all 4 training jobs complete
#
# Usage (from repo root):
#   bash scripts/submit_strategy_sweep.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$REPO_DIR/logs/slurm"

TRAIN_JID=$(sbatch --parsable --chdir="$REPO_DIR" "$REPO_DIR/scripts/sbatch_train_sweep.sh")
echo "Submitted training array job: $TRAIN_JID (strategies: 0322, 0324, mar30, apr08)"

CLS_JID=$(sbatch --parsable --chdir="$REPO_DIR" \
  --dependency=afterok:${TRAIN_JID} \
  "$REPO_DIR/scripts/sbatch_cls_vis_sweep.sh")
echo "Submitted cls+vis job:        $CLS_JID (starts after all training complete)"

echo ""
echo "Monitor with:  squeue -u \$USER"
echo "Training logs: $REPO_DIR/logs/slurm/train_${TRAIN_JID}_[0-3].out"
echo "Cls+vis log:   $REPO_DIR/logs/slurm/cls_vis_${CLS_JID}.out"
