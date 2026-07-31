REMOTE=liyading@login.ds.uchicago.edu
LOCAL=/home/lding/lding/dsicluster_CLS_rsync_folder/data/fa_data_analysis/ae_results
SRC=/net/projects/CLS/lding/data/fa_data_analysis/ae_results/test_run_overfit_20260322

mkdir -p $LOCAL/test_run_overfit_20260322/{baseline,semisup_fa,semisup_pos,semisup_both}

rsync -avh --progress \
    $REMOTE:$SRC/data.h5 \
    $LOCAL/test_run_overfit_20260322/

for m in baseline semisup_fa semisup_pos semisup_both; do
    rsync -avh --progress \
        $REMOTE:$SRC/$m/model.h5 \
        $LOCAL/test_run_overfit_20260322/$m/
done