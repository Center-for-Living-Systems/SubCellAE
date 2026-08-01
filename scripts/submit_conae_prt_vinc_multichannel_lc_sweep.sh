#!/usr/bin/env bash
# submit_conae_prt_vinc_multichannel_lc_sweep.sh
# ConAE + SupCon vinc with cio_mode_prt — 18 jobs (array 0-17)

set -eo pipefail
mkdir -p logs/slurm

echo "Submitting conae/supcon prt vinc sweep (18 configs)"

JOB=$(sbatch --parsable \
    --array=0-17 \
    scripts/sbatch_conae_prt_vinc_multichannel_lc_sweep.sh)
echo "Array job: $JOB  (tasks 0–17)"
