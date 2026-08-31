#!/usr/bin/env bash
# rsync_tif_to_local.sh
#
# Run this script FROM YOUR LOCAL MACHINE to pull all .tif reconstruction
# files from contrastive_* and ds_combo* runs on the cluster.
#
# Usage:
#   chmod +x rsync_tif_to_local.sh
#   ./rsync_tif_to_local.sh
#
# Adjust CLUSTER_HOST and LOCAL_DEST before running.

CLUSTER_HOST="liyading@<cluster-hostname>"   # e.g. liyading@midway3.rcc.uchicago.edu
CLUSTER_SRC="/net/projects/CLS/lding/data/fa_data_analysis/ae_results/contrastive_run"
LOCAL_DEST="$HOME/fa_data/contrastive_run_tifs"   # change to your preferred local path

mkdir -p "$LOCAL_DEST"

echo "Syncing .tif files from contrastive_* and ds_combo* ..."
echo "Source : ${CLUSTER_HOST}:${CLUSTER_SRC}"
echo "Dest   : ${LOCAL_DEST}"
echo ""

# contrastive_* runs
rsync -avz --progress \
    --include="*/" \
    --include="*.tif" \
    --exclude="*" \
    --filter="+ contrastive_*/" \
    --filter="- le_*/" \
    --filter="- supcon_*/" \
    "${CLUSTER_HOST}:${CLUSTER_SRC}/" \
    "${LOCAL_DEST}/" \
    --prune-empty-dirs

echo ""
echo "Syncing .tif files from ds_combo* ..."

# ds_combo* runs (separate pass for clarity)
rsync -avz --progress \
    --include="*/" \
    --include="*.tif" \
    --exclude="*" \
    "${CLUSTER_HOST}:${CLUSTER_SRC}/ds_combo_enlcrop_sc2_clip02_l1/" \
    "${LOCAL_DEST}/ds_combo_enlcrop_sc2_clip02_l1/"

rsync -avz --progress \
    --include="*/" \
    --include="*.tif" \
    --exclude="*" \
    "${CLUSTER_HOST}:${CLUSTER_SRC}/ds_combo_enlcrop_clip01_l1/" \
    "${LOCAL_DEST}/ds_combo_enlcrop_clip01_l1/"

rsync -avz --progress \
    --include="*/" \
    --include="*.tif" \
    --exclude="*" \
    "${CLUSTER_HOST}:${CLUSTER_SRC}/ds_combo_enlcrop_prt_sc2_l1/" \
    "${LOCAL_DEST}/ds_combo_enlcrop_prt_l1/"

rsync -avz --progress \
    --include="*/" \
    --include="*.tif" \
    --exclude="*" \
    "${CLUSTER_HOST}:${CLUSTER_SRC}/ds_combo_enlcrop_prt_l1/" \
    "${LOCAL_DEST}/ds_combo_enlcrop_prt_l1/"

echo ""
echo "Done. Files saved to: ${LOCAL_DEST}"
echo "Approximate size to expect: ~197G"
