#!/usr/bin/env bash
# =============================================================================
# Extract vinculin channel (ch3, rb647) patches with cio_rb normalisation
# from the vinc training dataset. Produces patches at
#   ae_results/pax_ch_patch/cio_rb/vinc_ch3/{control,ycomp}/tiff_patches32/
#
# NOTE: If major_ch: 3 is wrong for your CZI file layout, edit
#   config/screening_config/patchprep_vinc_ch3_control.yaml  (and ycomp)
#   to set the correct channel index before submitting.
#
# Submit: sbatch scripts/sbatch_extract_vinc_ch3.sh
# =============================================================================

#SBATCH --job-name=vinc_ch3_patchprep
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/vinc_ch3_patchprep_%j.out
#SBATCH --error=logs/vinc_ch3_patchprep_%j.err

set -euo pipefail

REPO="/net/projects/CLS/lding/gitcode/SubCellAE-nonad-vs-ad"
cd "$REPO"
mkdir -p logs

set +u
source /home/liyading/miniconda3/etc/profile.d/conda.sh
conda activate subcellae-cuda
set -u

PYTHON=$(which python)

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "Start:  $(date)"
echo ""

echo "=== Extracting vinculin (ch3) patches — control ==="
$PYTHON scripts/run_patchprep_from_config.py \
    config/screening_config/patchprep_vinc_ch3_control.yaml

echo ""
echo "=== Extracting vinculin (ch3) patches — ycomp ==="
$PYTHON scripts/run_patchprep_from_config.py \
    config/screening_config/patchprep_vinc_ch3_ycomp.yaml

echo ""
echo "Done: $(date)"
