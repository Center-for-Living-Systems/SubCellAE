#!/usr/bin/env bash
# submit_le_clean.sh — generate configs and submit SupCon AE array for the
# clean label-efficiency experiment.
#
# Usage:
#   bash scripts/submit_le_clean.sh            # setup + train
#   bash scripts/submit_le_clean.sh --dry-run  # preview only

set -eo pipefail

REPO="$PWD"
PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
export PYTHONPATH="$REPO"
mkdir -p logs/slurm

DRY=""
for arg in "$@"; do
    [[ "$arg" == "--dry-run" ]] && DRY="--dry-run"
done

# ── Step 1: generate annotation CSVs + YAML configs ──────────────────────────
echo "=== Generating configs ==="
$PYTHON scripts/setup_label_efficiency_clean.py $DRY

if [[ -n "$DRY" ]]; then
    echo "[dry-run] Stopping before submission."
    exit 0
fi

# ── Step 2: submit SLURM array ───────────────────────────────────────────────
N=$(wc -l < config/label_efficiency/job_list.txt)
echo ""
echo "=== Submitting array ($N jobs) ==="

JOB_ID=$(sbatch --parsable \
    --array=0-$((N - 1)) \
    scripts/sbatch_le_clean_array.sh)

echo "Array job ID : $JOB_ID  (tasks 0–$((N - 1)))"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f logs/slurm/le_clean_${JOB_ID}_0.out"
echo ""
echo "After all jobs finish, run the evaluation:"
echo "  python scripts/run_label_efficiency_clean.py"
