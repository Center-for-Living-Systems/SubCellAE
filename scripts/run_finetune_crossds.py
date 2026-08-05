#!/usr/bin/env python3
"""
run_finetune_crossds.py

Fine-tune the supcon2 2-class model (trained on vinc/control) by adding
ppax/control and pfak/control labeled patches from:
  labels_ppax_20260521.csv  →  ppax_control_label_2cls.csv  (51 patches, excl. Uncertain)
  labels_pfak_20260521.csv  →  pfak_control_label_2cls.csv  (54 patches)

Starting from model_best.pt for the chosen split, trains for --epochs epochs
at --lr with:
  - All vinc/control 32×32 patches  (~14k)  + Annabel's 2-cls labels
  - All ppax/control 32×32 patches  (~2.9k) + 2-cls labels
  - All pfak/control 32×32 patches  (~2.8k) + 2-cls labels

Result saved to:
  {contrastive_run}/annabel_vinc_ppax_pfak_supcon2_ft_{split}/

Usage:
  python scripts/run_finetune_crossds.py [--split s2v2] [--epochs 200] [--lr 2e-4]
  python scripts/run_finetune_crossds.py --split s1v3
  python scripts/run_finetune_crossds.py --split s3v1
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
    ap.add_argument("--epochs", type=int,   default=200)
    ap.add_argument("--lr",     type=float, default=2e-4)
    ap.add_argument("--warmup", type=int,   default=0)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--labeled-only", action="store_true",
                    help="Train only on annotated patches (644 total). Much faster on CPU.")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    base_dir   = RUN_DIR / f"annabel_vinc_supcon2_{args.split}"
    ckpt_path  = base_dir / "model_best.pt"
    suffix     = "ft_labeled" if args.labeled_only else "ft"
    result_dir = RUN_DIR / f"annabel_vinc_ppax_pfak_supcon2_{suffix}_{args.split}"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Base checkpoint not found: {ckpt_path}")

    mode_str = "labeled-only (644 patches)" if args.labeled_only else "full (20k patches)"
    print(f"Fine-tuning from: {ckpt_path}")
    print(f"Result dir      : {result_dir}")
    print(f"Mode            : {mode_str}")
    print(f"Epochs={args.epochs}  LR={args.lr}  Warmup={args.warmup}")

    from subcellae.pipeline.ae_pipeline import AEConfig, run_ae_pipeline

    cfg = AEConfig(
        result_dir=result_dir,

        # ── Three patch directories, each with its own annotation file ─────
        patch_dirs=[
            {
                "path":           str(PATCH_BASE / "cio/vinc/control/tiff_patches32_mr10"),
                "condition":      0,
                "condition_name": "vinc_control",
                "annotation_file": str(LABEL_DIR / "vinc_control_label_Annabel_20260715_1554_2cls.csv"),
                "label_col":      "label",
                "filename_col":   "unique_ID",    # hyphen format → matches _patch_name_to_annotation_key
                "label_order":    LABEL_ORDER_2CLS,
            },
            {
                "path":           str(PATCH_BASE / "cio_rb/ppax/control/tiff_patches32_mr10"),
                "condition":      1,
                "condition_name": "ppax_control",
                "annotation_file": str(LABEL_DIR / "ppax_control_label_2cls.csv"),
                "label_col":      "label",
                "filename_col":   "unique_ID",
                "label_order":    LABEL_ORDER_2CLS,
            },
            {
                "path":           str(PATCH_BASE / "cio_rb/pfak/control/tiff_patches32_mr10"),
                "condition":      2,
                "condition_name": "pfak_control",
                "annotation_file": str(LABEL_DIR / "pfak_control_label_2cls.csv"),
                "label_col":      "label",
                "filename_col":   "unique_ID",
                "label_order":    LABEL_ORDER_2CLS,
            },
        ],

        # ── Model (same architecture as base) ────────────────────────────
        model_type      = "supcon",
        latent_dim      = 12,
        proj_dim        = 8,
        input_ps        = 32,
        no_ch           = 1,
        BN_flag         = False,
        dropout_flag    = False,
        output_sigmoid  = False,
        recon_loss_type = "nl1",

        # ── Supcon loss params (match base) ──────────────────────────────
        noise_prob            = 0.0,
        temperature           = 0.5,
        lambda_recon          = 1.0,
        lambda_contrast       = 0.5,
        intensity_scale_range = (0.8, 1.2),

        # ── Fine-tuning hypers ───────────────────────────────────────────
        pretrained_checkpoint = str(ckpt_path),
        labeled_only          = args.labeled_only,
        epochs                = args.epochs,
        lr                    = args.lr,
        batch_size            = args.batch_size,
        num_workers           = 0 if args.labeled_only else 4,
        val_split             = 0.15 if args.labeled_only else 0.2,
        group_split           = not args.labeled_only,
        weight_decay          = 1e-4,
        warmup_epochs         = args.warmup,
        lr_scheduler          = "none",
        early_stopping_patience = 0,
        min_epochs_for_best   = 0,

        # ── Reconstruction output ────────────────────────────────────────
        save_recon       = True,
        recon_pad_size   = 64,
        recon_image_size = 1024,

        device = "auto",
    )

    run_ae_pipeline(cfg)
    print(f"\nDone. Result: {result_dir}")


if __name__ == "__main__":
    main()
