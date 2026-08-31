#!/usr/bin/env python3
"""
run_supcon_corrected.py

Re-run Stage 1 SupCon (binary labels: no-adh vs adh) on control-only patches
with the CORRECT filename_col="unique_ID" so labels actually match.

Previous Stage 1 runs (annabel_vinc_supcon2_s1v3/s2v2/s3v1) used
filename_col="filename" (underscore format) which produced 0 label matches —
those runs were effectively ConAE. These corrected runs use filename_col="unique_ID"
(hyphen format matching _patch_name_to_annotation_key).

Output:
  {RUN_DIR}/annabel_vinc_supcon2_corrected_{split}/

Usage:
  python scripts/run_supcon_corrected.py [--split s2v2] [--all-splits]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DATA_ROOT   = Path("/net/projects/CLS/lding/data/fa_data_analysis")
RUN_DIR     = DATA_ROOT / "ae_results" / "contrastive_run"
PATCH_DIR   = DATA_ROOT / "ae_results" / "patches" / "cio/vinc/control/tiff_patches32_mr10"
LABEL_DIR   = DATA_ROOT / "labelling"

SPLIT_VAL_FRAC = {"s1v3": 0.75, "s2v2": 0.50, "s3v1": 0.25}


def run_split(split: str):
    from subcellae.pipeline.ae_pipeline import AEConfig, run_ae_pipeline

    val_frac   = SPLIT_VAL_FRAC[split]
    result_dir = RUN_DIR / f"annabel_vinc_supcon2_corrected_{split}"

    print(f"\n{'='*65}")
    print(f"SupCon corrected  split={split}  val_frac={val_frac}")
    print(f"Output: {result_dir}")
    print(f"{'='*65}")

    cfg = AEConfig(
        result_dir=result_dir,

        patch_dirs=[{
            "path":            str(PATCH_DIR),
            "condition":       0,
            "condition_name":  "control",
            "annotation_file": str(LABEL_DIR / "vinc_control_label_Annabel_20260715_1554_2cls.csv"),
            "label_col":       "label",
            "filename_col":    "unique_ID",      # correct: hyphen format matches _patch_name_to_annotation_key
            "label_order":     ["No adhesion", "adhesion"],
        }],

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
        val_split               = val_frac,
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
    print(f"\nDone: {result_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="s2v2", choices=list(SPLIT_VAL_FRAC))
    ap.add_argument("--all-splits", action="store_true")
    args = ap.parse_args()

    splits = list(SPLIT_VAL_FRAC) if args.all_splits else [args.split]
    for sp in splits:
        run_split(sp)
    print("\nAll done.")


if __name__ == "__main__":
    main()
