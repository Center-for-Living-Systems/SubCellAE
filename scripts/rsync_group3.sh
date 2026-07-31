REMOTE=liyading@login.ds.uchicago.edu
LOCAL=/home/lding/lding/dsicluster_CLS_rsync_folder/data/fa_data_analysis/ae_results
SRC=/net/projects/CLS/lding/data/fa_data_analysis/ae_results/contrastive_run

# Flat models (contrastive_cio_*, supcon_cio_*)
rsync -avh --progress \
    --include="*/" \
    --include="model.h5" \
    --exclude="*" \
    $REMOTE:$SRC/ \
    $LOCAL/contrastive_run/

# ds_combo nested models (ds_combo_*/combo_name/model.h5)
for parent in ds_combo_enlcrop_clip01_l1 ds_combo_enlcrop_sc2_clip02_l1 \
              ds_combo_enlcrop_sc2 ds_combo_enlcrop_sc2_lc010_bal \
              ds_combo_enlcrop_sc2_lc010_bal_l1 ds_combo_enlcrop_sc2_lc010_bal_mse; do
    rsync -avh --progress \
        --include="*/" \
        --include="model.h5" \
        --exclude="*" \
        $REMOTE:$SRC/$parent/ \
        $LOCAL/contrastive_run/$parent/
done