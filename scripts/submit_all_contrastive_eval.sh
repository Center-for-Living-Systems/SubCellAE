#!/usr/bin/env bash
# Submit one contrastive eval job per result dir that has latents.csv but no eval/ yet.
# Re-run any time to pick up newly completed training runs.
#
# Usage (from repo root):
#   bash scripts/submit_all_contrastive_eval.sh            # skip dirs with eval/ already
#   bash scripts/submit_all_contrastive_eval.sh --rerun    # resubmit all, even if eval/ exists

set -eo pipefail

BASE="/net/projects/CLS/lding/data/fa_data_analysis/ae_results/contrastive_run"
RERUN=0
[[ "${1:-}" == "--rerun" ]] && RERUN=1

submitted=0
skipped=0

for d in "$BASE"/*/; do
    [[ -d "$d" ]] || continue
    name=$(basename "$d")

    # Must have latents.csv (training complete)
    if [[ ! -f "$d/latents.csv" ]]; then
        echo "SKIP (no latents.csv): $name"
        ((skipped++)) || true
        continue
    fi

    # Skip if eval completed successfully (summary exists), unless --rerun
    if [[ -f "$d/eval/eval_summary.csv" && $RERUN -eq 0 ]]; then
        echo "SKIP (eval complete):  $name"
        ((skipped++)) || true
        continue
    fi

    jid=$(sbatch --parsable scripts/sbatch_contrastive_eval.sh "$d")
    echo "SUBMITTED [$jid]: $name"
    ((submitted++)) || true
done

echo ""
echo "Done. Submitted: $submitted  |  Skipped: $skipped"
