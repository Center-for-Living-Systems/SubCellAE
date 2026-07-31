REMOTE=liyading@login.ds.uchicago.edu
LOCAL=/home/lding/lding/dsicluster_CLS_rsync_folder/data/fa_data_analysis/ae_results

for ds in vinc ppax pfak nih3t3; do
    mkdir -p $LOCAL/patches/cio/$ds
    rsync -avh --progress \
        $REMOTE:/net/projects/CLS/lding/data/fa_data_analysis/ae_results/patches/cio/$ds/data.h5 \
        $LOCAL/patches/cio/$ds/
done