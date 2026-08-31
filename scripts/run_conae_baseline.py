#!/usr/bin/env python3
"""
run_conae_baseline.py

Train a plain ConAE (NT-Xent contrastive, no labels) on all vinc/control patches
as a baseline to compare against Stage 1 SupCon.

Same architecture and hyperparams as Stage 1 SupCon s2v2, but model_type="contrastive"
and no annotation file — every patch is treated as unlabeled (self-augmentation pairs only).

Since there are no labels, train/val split only affects reconstruction eval;
we use s2v2-equivalent val_split=0.5 for comparability.

Output:
  {RUN_DIR}/annabel_vinc_conae/
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DATA_ROOT  = Path("/net/projects/CLS/lding/data/fa_data_analysis")
RUN_DIR    = DATA_ROOT / "ae_results" / "contrastive_run"
PATCH_DIR  = DATA_ROOT / "ae_results" / "patches" / "cio/vinc/control/tiff_patches32_mr10"


def main():
    from subcellae.pipeline.ae_pipeline import AEConfig, run_ae_pipeline

    result_dir = RUN_DIR / "annabel_vinc_conae"

    print(f"ConAE baseline — all vinc/control patches, no labels")
    print(f"Output: {result_dir}")

    cfg = AEConfig(
        result_dir=result_dir,

        patch_dirs=[{
            "path":           str(PATCH_DIR),
            "condition":      0,
            "condition_name": "vinc_control",
        }],

        model_type      = "contrastive",
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

        epochs                  = 500,
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
    print(f"\nConAE done: {result_dir}")


if __name__ == "__main__":
    main()
