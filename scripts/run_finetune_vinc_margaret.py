#!/usr/bin/env python3
"""
run_finetune_vinc_margaret.py

Fine-tune the supcon2 2-class model (trained on vinc/control Annabel only)
by adding Margaret's vinc/control labels.

Starting from model_best.pt for the chosen split, trains for --epochs epochs
using ONLY vinc/control patches with Annabel + Margaret combined labels (899 patches).

No ppax, pfak, or nih3t3 data is used.

Result saved to:
  {contrastive_run}/annabel_vinc_margaret_ft_labeled_{split}/

Usage:
  python scripts/run_finetune_vinc_margaret.py [--split s2v2] [--epochs 50] [--lr 2e-4]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DATA_ROOT  = Path("/net/projects/CLS/lding/data/fa_data_analysis")
LABEL_DIR  = DATA_ROOT / "labelling"
RUN_DIR    = DATA_ROOT / "ae_results" / "contrastive_run"
PATCH_BASE = DATA_ROOT / "ae_results" / "patches"

LABEL_ORDER_2CLS = ["No adhesion", "adhesion"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split",  default="s2v2",
                    choices=["s1v3", "s2v2", "s3v1"])
    ap.add_argument("--epochs", type=int,   default=50)
    ap.add_argument("--lr",     type=float, default=2e-4)
    ap.add_argument("--warmup", type=int,   default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    base_dir   = RUN_DIR / f"annabel_vinc_supcon2_{args.split}"
    ckpt_path  = base_dir / "model_best.pt"
    result_dir = RUN_DIR / f"annabel_vinc_margaret_ft_labeled_{args.split}"
    label_csv  = LABEL_DIR / "vinc_control_label_combined_2cls.csv"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Base checkpoint not found: {ckpt_path}")
    if not label_csv.exists():
        raise FileNotFoundError(f"Combined label CSV not found: {label_csv}")

    print(f"Fine-tuning from : {ckpt_path}")
    print(f"Result dir       : {result_dir}")
    print(f"Label CSV        : {label_csv}  (Annabel + Margaret, 899 patches)")
    print(f"Epochs={args.epochs}  LR={args.lr}  Warmup={args.warmup}")
    print("Mode: labeled-only (training only on 899 annotated vinc patches)")

    from subcellae.pipeline.ae_pipeline import AEConfig, run_ae_pipeline

    cfg = AEConfig(
        result_dir=result_dir,

        # vinc/control patches with combined Annabel+Margaret labels
        patch_dirs=[
            {
                "path":            str(PATCH_BASE / "cio/vinc/control/tiff_patches32_mr10"),
                "condition":       0,
                "condition_name":  "vinc_control",
                "annotation_file": str(label_csv),
                "label_col":       "label",
                "filename_col":    "unique_ID",
                "label_order":     LABEL_ORDER_2CLS,
            },
        ],

        # Model architecture — same as base
        model_type      = "supcon",
        latent_dim      = 12,
        proj_dim        = 8,
        input_ps        = 32,
        no_ch           = 1,
        BN_flag         = False,
        dropout_flag    = False,
        output_sigmoid  = False,
        recon_loss_type = "nl1",

        # Supcon loss params — match base
        noise_prob            = 0.0,
        temperature           = 0.5,
        lambda_recon          = 1.0,
        lambda_contrast       = 0.5,
        intensity_scale_range = (0.8, 1.2),

        # Fine-tuning hypers
        pretrained_checkpoint   = str(ckpt_path),
        labeled_only            = True,
        epochs                  = args.epochs,
        lr                      = args.lr,
        batch_size              = args.batch_size,
        num_workers             = 0,
        val_split               = 0.15,
        group_split             = False,
        weight_decay            = 1e-4,
        warmup_epochs           = args.warmup,
        lr_scheduler            = "none",
        early_stopping_patience = 0,
        min_epochs_for_best     = 0,

        # Reconstruction output
        save_recon       = True,
        recon_pad_size   = 64,
        recon_image_size = 1024,

        device = "auto",
    )

    run_ae_pipeline(cfg)
    print(f"\nDone. Result: {result_dir}")


if __name__ == "__main__":
    main()
