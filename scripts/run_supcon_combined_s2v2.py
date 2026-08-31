#!/usr/bin/env python3
"""
run_supcon_combined_s2v2.py

SupCon (binary labels) on control + ycomp patches combined, s2v2 split.

Control frames 0-1 → train (labeled binary: no-adh vs adh).
Control frames 2-3 → val. ycomp patches all unlabeled.
Per-dataset val_split ensures control frames get the same s2v2 assignment
regardless of ycomp pooling.

Compare against annabel_vinc_supcon2_s2v2 (control-only) to see if adding
ycomp changes the latent space / UMAP structure.

Output:
  {RUN_DIR}/annabel_vinc_supcon2_combined_s2v2/
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DATA_ROOT   = Path("/net/projects/CLS/lding/data/fa_data_analysis")
RUN_DIR     = DATA_ROOT / "ae_results" / "contrastive_run"
CONTROL_DIR = DATA_ROOT / "ae_results" / "patches" / "cio/vinc/control/tiff_patches32_mr10"
YCOMP_DIR   = DATA_ROOT / "ae_results" / "patches" / "cio/vinc/ycomp/tiff_patches32_mr10"
LABEL_DIR   = DATA_ROOT / "labelling"


def main():
    from subcellae.pipeline.ae_pipeline import AEConfig, run_ae_pipeline

    result_dir = RUN_DIR / "annabel_vinc_supcon2_combined_s2v2"
    print(f"SupCon s2v2 combined (control + ycomp)")
    print(f"Output: {result_dir}")

    cfg = AEConfig(
        result_dir=result_dir,

        patch_dirs=[
            {
                "path":            str(CONTROL_DIR),
                "condition":       0,
                "condition_name":  "control",
                "annotation_file": str(LABEL_DIR / "vinc_control_label_Annabel_20260715_1554_2cls.csv"),
                "label_col":       "label",
                "filename_col":    "unique_ID",
                "label_order":     ["No adhesion", "adhesion"],
                "val_split":       0.5,   # per-dataset → same s2v2 frame split as control-only
            },
            {
                "path":           str(YCOMP_DIR),
                "condition":      1,
                "condition_name": "ycomp",
                "val_split":      0.5,   # ycomp unlabeled; 50% to val for recon eval
            },
        ],

        model_type      = "supcon",
        latent_dim      = 12,
        proj_dim        = 8,
        input_ps        = 32,
        no_ch           = 1,
        BN_flag         = False,
        dropout_flag    = False,
        output_sigmoid  = False,
        recon_loss_type = "nl1",

        noise_prob            = 0.0,
        temperature           = 0.5,
        lambda_recon          = 1.0,
        lambda_contrast       = 0.5,
        intensity_scale_range = (0.8, 1.2),

        epochs                  = 300,
        lr                      = 1e-3,
        batch_size              = 128,
        num_workers             = 0,
        val_split               = 0.5,
        group_split             = True,
        weight_decay            = 1e-4,
        warmup_epochs           = 0,
        lr_scheduler            = "none",
        early_stopping_patience = 0,
        min_epochs_for_best     = 0,

        save_recon       = True,
        recon_pad_size   = 64,
        recon_image_size = 1024,

        device = "auto",
    )

    run_ae_pipeline(cfg)
    print(f"\nSupCon combined done: {result_dir}")


if __name__ == "__main__":
    main()
