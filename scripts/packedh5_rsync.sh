for dataset in vinc ppax pfak nih3t3
do
    mkdir -p /home/lding/lding/dsicluster_CLS_rsync_folder/data/fa_data_analysis/ae_results/patches/cio_rb/${dataset}

    mkdir -p /mnt/p/Liya/FA_patch_group_label/${dataset}

    rsync -avh \
    liyading@login.ds.uchicago.edu:/net/projects/CLS/lding/data/fa_data_analysis/ae_results/patches/cio_rb/${dataset}/${dataset}_control_label.h5 \
    liyading@login.ds.uchicago.edu:/net/projects/CLS/lding/data/fa_data_analysis/ae_results/patches/cio_rb/${dataset}/${dataset}_ycomp_label.h5 \
    /home/lding/lding/dsicluster_CLS_rsync_folder/data/fa_data_analysis/ae_results/patches/cio_rb/${dataset}/

    rsync -avh \
    liyading@login.ds.uchicago.edu:/net/projects/CLS/lding/data/fa_data_analysis/ae_results/patches/cio_rb/${dataset}/${dataset}_control_label.h5 \
    liyading@login.ds.uchicago.edu:/net/projects/CLS/lding/data/fa_data_analysis/ae_results/patches/cio_rb/${dataset}/${dataset}_ycomp_label.h5 \
    /mnt/p/Liya/FA_patch_group_label/${dataset}/
done