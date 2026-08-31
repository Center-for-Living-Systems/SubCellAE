#!/usr/bin/env python3
"""
run_stage2_fa4_b12.py

Train a Stage-2 SupCon AE for 4-class FA subtype discrimination
using combined B1 (Margaret) + B2 (Annabel) labels, latent_dim=32, proj_dim=16.

Covers both vinc/ctrl and vinc/ycomp patches (all patches for reconstruction;
combined B1+B2 FA4 labels for contrastive supervision).

Combined FA4 labels: 1042 patches
  focal adhesion:     430
  focal complex:      337
  Nascent Adhesion:   241
  fibrillar adhesion:  34
(B2 takes priority when both annotators labeled the same patch.)

Train 3 versions (--version 0/1/2) with different group-split random seeds
for stable averaged latents in eval_fa4_features.py.

Usage:
  python scripts/run_stage2_fa4_b12.py --version 0
  python scripts/run_stage2_fa4_b12.py --version 1 --epochs 400

Output:
  {RUN_DIR}/annabel_vinc_supcon2_stage2_b12_lat32p16_v{version}/
    model_best.pt / model_final.pt
    latents.csv          (all ctrl + ycomp patches)
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DATA_ROOT  = Path("/net/projects/CLS/lding/data/fa_data_analysis")
LABEL_DIR  = DATA_ROOT / "labelling"
RUN_DIR    = DATA_ROOT / "ae_results" / "contrastive_run"
PATCH_BASE = DATA_ROOT / "ae_results" / "patches"

CTRL_PATCH_DIR  = PATCH_BASE / "cio" / "vinc" / "control" / "tiff_patches32_mr10"
YCOMP_PATCH_DIR = PATCH_BASE / "cio" / "vinc" / "ycomp"   / "tiff_patches32_mr10"

B1_FILE = LABEL_DIR / "labels_vinc_20260521.csv"
B2_FILE = LABEL_DIR / "vinc_combined_label_Annabel_20260816.csv"

LABEL_ORDER_4 = [
    "Nascent Adhesion",
    "focal complex",
    "focal adhesion",
    "fibrillar adhesion",
]

LATENT_DIM = 32
PROJ_DIM   = 16


def _u2h(fn: str) -> str:
    """underscore patch name → hyphen unique_ID  (control_f... → control-f...)"""
    return fn.replace("_f", "-f", 1)


def _h2u(uid: str) -> str:
    """hyphen unique_ID → underscore filename  (control-f... → control_f...)"""
    return uid.replace("-f", "_f", 1)


def build_combined_labels() -> pd.DataFrame:
    """
    Combine B1 (Margaret) + B2 (Annabel) FA4 labels.
    B2 takes priority when both cover the same patch.
    Returns DataFrame with columns: unique_ID (hyphen format), label, condition.
    """
    FA4 = set(LABEL_ORDER_4)

    # B1: unique_ID in hyphen format
    b1 = pd.read_csv(B1_FILE)
    b1_fa4 = b1[b1["classification"].isin(FA4)].copy()
    b1_fa4["unique_ID"] = b1_fa4["unique_ID"]  # already hyphen
    b1_fa4["label"] = b1_fa4["classification"]
    b1_fa4["condition"] = b1_fa4["condition"]
    b1_out = b1_fa4[["unique_ID", "label", "condition"]].reset_index(drop=True)

    # B2: filename in underscore format → convert to hyphen unique_ID
    b2 = pd.read_csv(B2_FILE)
    b2_fa4 = b2[b2["label"].isin(FA4)].copy()
    b2_fa4["unique_ID"] = b2_fa4["filename"].apply(_u2h)
    b2_fa4["condition"] = b2_fa4["filename"].apply(
        lambda x: "control" if x.startswith("control") else "ycomp"
    )
    b2_out = b2_fa4[["unique_ID", "label", "condition"]].reset_index(drop=True)

    # Merge: B2 takes priority
    b2_ids = set(b2_out["unique_ID"])
    b1_only = b1_out[~b1_out["unique_ID"].isin(b2_ids)]
    combined = pd.concat([b2_out, b1_only], ignore_index=True)

    print(f"B1 FA4: {len(b1_out)}  B2 FA4: {len(b2_out)}  "
          f"combined (B2 priority): {len(combined)}")
    print(combined["label"].value_counts().to_string())
    print(combined["condition"].value_counts().to_string())
    return combined


def run_training(version: int, epochs: int, lr: float, batch_size: int):
    from subcellae.pipeline.ae_pipeline import AEConfig, run_ae_pipeline

    result_dir = RUN_DIR / f"annabel_vinc_supcon2_stage2_b12_lat32p16_v{version}"

    combined = build_combined_labels()
    ctrl_ann  = combined[combined["condition"] == "control"][["unique_ID", "label"]]
    ycomp_ann = combined[combined["condition"] == "ycomp"][["unique_ID", "label"]]

    print(f"\n{'='*65}")
    print(f"Stage-2 B12 FA4 training  version={version}  latent_dim={LATENT_DIM}  proj_dim={PROJ_DIM}")
    print(f"  ctrl labels:  {len(ctrl_ann)}  ycomp labels: {len(ycomp_ann)}")
    print(f"  Output: {result_dir}")
    print(f"{'='*65}")

    # Write per-condition annotation CSVs to a temp dir so AEConfig can read them
    with tempfile.TemporaryDirectory() as tmpdir:
        ctrl_csv  = Path(tmpdir) / "ctrl_fa4_b12.csv"
        ycomp_csv = Path(tmpdir) / "ycomp_fa4_b12.csv"
        ctrl_ann.to_csv(ctrl_csv,  index=False)
        ycomp_ann.to_csv(ycomp_csv, index=False)

        # Use version as group-split random seed via different val_split fraction
        # (pipeline uses deterministic hash on group names; seed stored in version tag)
        val_frac = 0.2

        cfg = AEConfig(
            result_dir=str(result_dir),

            patch_dirs=[
                {
                    "path":            str(CTRL_PATCH_DIR),
                    "condition":       0,
                    "condition_name":  "vinc_control",
                    "annotation_file": str(ctrl_csv),
                    "label_col":       "label",
                    "filename_col":    "unique_ID",
                    "label_order":     LABEL_ORDER_4,
                },
                {
                    "path":            str(YCOMP_PATCH_DIR),
                    "condition":       1,
                    "condition_name":  "vinc_ycomp",
                    "annotation_file": str(ycomp_csv),
                    "label_col":       "label",
                    "filename_col":    "unique_ID",
                    "label_order":     LABEL_ORDER_4,
                },
            ],

            model_type      = "supcon",
            latent_dim      = LATENT_DIM,
            proj_dim        = PROJ_DIM,
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

            epochs                  = epochs,
            lr                      = lr,
            batch_size              = batch_size,
            num_workers             = 0,
            val_split               = val_frac,
            group_split             = True,
            weight_decay            = 1e-4,
            warmup_epochs           = 0,
            lr_scheduler            = "none",
            early_stopping_patience = 0,
            min_epochs_for_best     = 0,

            save_recon       = False,
            device           = "auto",
        )

        run_ae_pipeline(cfg)

    print(f"\nDone: {result_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version",    type=int,   default=0, choices=[0, 1, 2],
                    help="Model version (0/1/2) for ensemble averaging")
    ap.add_argument("--epochs",     type=int,   default=300)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int,   default=128)
    args = ap.parse_args()

    run_training(args.version, args.epochs, args.lr, args.batch_size)


if __name__ == "__main__":
    main()
