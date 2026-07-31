#!/bin/bash

REMOTE="liyading@login.ds.uchicago.edu:/net/projects/CLS/lding/data/fa_data_analysis/ae_results/patches"
LOCAL="/mnt/p/image_service_data/FA_patch_data"

for dataset in vinc pfak ppax nih3t3; do
    for type in cio cio_rb; do
        mkdir -p "/mnt/p/image_service_data/FA_patch_data/$type/$dataset"

        rsync -avh \
            --partial \
            --append-verify \
            --info=progress2,stats2 \
            --timeout=300 \
            "$REMOTE/$type/$dataset/data.h5" \
            "/mnt/p/image_service_data/FA_patch_data/$type/$dataset/"
    done
done