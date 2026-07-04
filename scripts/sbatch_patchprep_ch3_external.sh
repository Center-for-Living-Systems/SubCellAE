#!/usr/bin/env bash
#SBATCH --job-name=patchprep_ch3_ext
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/patchprep_ch3_external_%j.out

# Extract ch3 (actin/phalloidin) patches for ppax, pfak, nih3t3
# using the same FA positions as the existing ch1 patches.

set -eo pipefail
exec 2>&1

REPO="$PWD"
export PYTHONPATH="$REPO"
mkdir -p logs/slurm

echo "======================================================================"
echo "Job:   $SLURM_JOB_ID"
echo "Node:  $(hostname)"
echo "Start: $(date)"
echo "======================================================================"

CFG="config/patchprep_config"

CONFIGS=(
    "$CFG/ppax_control_cio_rb_ch3.yaml"
    "$CFG/ppax_ycomp_cio_rb_ch3.yaml"
    "$CFG/pfak_control_cio_rb_ch3.yaml"
    "$CFG/pfak_ycomp_cio_rb_ch3.yaml"
    "$CFG/nih3t3_control_cio_rb_ch3.yaml"
    "$CFG/nih3t3_ycomp_cio_rb_ch3.yaml"
)

for cfg in "${CONFIGS[@]}"; do
    echo ""
    echo "--- Patchprep: $cfg ---"
    python scripts/run_patchprep_from_config.py "$cfg"
done

echo ""
echo "End: $(date)"
