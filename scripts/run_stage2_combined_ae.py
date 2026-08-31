#!/usr/bin/env python3
"""
run_stage2_combined_ae.py

Option B — Stage 2 combined SupCon AE for FA subtype classification.

Trains a dedicated Stage-2 AE on labeled FA patches from ALL datasets:
  vinc/ctrl  : Stage-1 predicted-adhesion patches (5771), 4-class Annabel labels
  vinc/ycomp : all ycomp patches, 4-class Annabel labels for labeled subset
  pfak/ctrl  : all pfak/ctrl patches, 4-class Annabel labels
  ppax/ctrl  : all ppax/ctrl patches, 4-class Ernest labels (FA subtypes only)

Architecture: same SupCon AE as Stage 1 (latent_dim=12, proj_dim=8, ps=32).
Pretrained checkpoint: annabel_vinc_supcon2_corrected_s3v1/model_best.pt
Output: RUN_DIR/annabel_vinc_supcon2_stage2_combined/

Usage:
  python scripts/run_stage2_combined_ae.py
  python scripts/run_stage2_combined_ae.py --epochs 300 --lr 5e-4
"""
from __future__ import annotations

import argparse
import re
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

PATCH_DIRS = {
    "vinc_ctrl":  PATCH_BASE / "cio"    / "vinc" / "control" / "tiff_patches32_mr10",
    "vinc_ycomp": PATCH_BASE / "cio"    / "vinc" / "ycomp"   / "tiff_patches32_mr10",
    "pfak_ctrl":  PATCH_BASE / "cio_rb" / "pfak" / "control" / "tiff_patches32_mr10",
    "ppax_ctrl":  PATCH_BASE / "cio"    / "ppax" / "control" / "tiff_patches32_mr10",
}

LABEL_ORDER_4 = [
    "Nascent Adhesion",
    "focal complex",
    "focal adhesion",
    "fibrillar adhesion",
]

STAGE1_GATE_DIR = RUN_DIR / "annabel_vinc_supcon2_s2v2"  # Stage-1 binary gate
PRETRAINED_CKPT = RUN_DIR / "annabel_vinc_supcon2_corrected_s3v1" / "model_best.pt"
RESULT_DIR      = RUN_DIR / "annabel_vinc_supcon2_stage2_combined"


def _unique_id(filename: str) -> str:
    """Convert underscore patch name to hyphen format for annotation key matching."""
    return re.sub(r"_(f\d+x\d+y\d+ps\d+\.tiff?)", r"-\1", filename)


def _make_ann_csv(df: pd.DataFrame, tmp_dir: str, name: str) -> str:
    """Write annotation CSV with filename, unique_ID, label columns."""
    df = df.copy()
    df["unique_ID"] = df["filename"].apply(_unique_id)
    out = Path(tmp_dir) / name
    df[["filename", "unique_ID", "label"]].to_csv(out, index=False)
    return str(out)


def _get_stage1_adhesion_filenames() -> list[str]:
    """Run Stage-1 binary GBM on vinc/ctrl latents; return predicted-adhesion filenames."""
    import joblib
    stage1_lat = pd.read_csv(STAGE1_GATE_DIR / "blind_test" / "vinc_control_latents.csv")
    z_cols = [c for c in stage1_lat.columns if c.startswith("z_")]
    model  = joblib.load(str(STAGE1_GATE_DIR / "fa_cls_zrecon" / "model.pkl"))
    preds  = model.predict(stage1_lat[z_cols].values)
    adh_fns = stage1_lat.loc[preds == 1, "filename"].tolist()
    print(f"Stage-1 gate: {len(adh_fns)}/{len(stage1_lat)} vinc/ctrl patches predicted adhesion")
    return adh_fns


def run(epochs: int, lr: float, batch_size: int):
    from subcellae.pipeline.ae_pipeline import AEConfig, run_ae_pipeline

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*65}")
    print(f"Stage-2 Combined AE training  (Option B)")
    print(f"Pretrained: {PRETRAINED_CKPT}")
    print(f"Output:     {RESULT_DIR}")
    print(f"epochs={epochs}  lr={lr}  batch_size={batch_size}")
    print(f"{'='*65}")

    adh_fns = _get_stage1_adhesion_filenames()

    with tempfile.TemporaryDirectory() as tmp:
        # ── vinc/ctrl labels ─────────────────────────────────────────────────
        ctrl_df = pd.read_csv(LABEL_DIR / "vinc_control_label_Annabel_20260715_1554.csv")
        ctrl_fa = ctrl_df[ctrl_df["label"].isin(LABEL_ORDER_4)][["filename", "label"]]
        ctrl_ann = _make_ann_csv(ctrl_fa, tmp, "ann_vinc_ctrl.csv")
        print(f"vinc/ctrl FA labels: {len(ctrl_fa)} patches")

        # ── vinc/ycomp labels ────────────────────────────────────────────────
        comb_df  = pd.read_csv(LABEL_DIR / "vinc_combined_label_Annabel_20260816.csv")
        ycomp_fa = comb_df[
            comb_df["filename"].str.startswith("ycomp_") &
            comb_df["label"].isin(LABEL_ORDER_4)
        ][["filename", "label"]].copy()
        ycomp_ann = _make_ann_csv(ycomp_fa, tmp, "ann_vinc_ycomp.csv")
        print(f"vinc/ycomp FA labels: {len(ycomp_fa)} patches")

        # ── pfak/ctrl labels ─────────────────────────────────────────────────
        pfak_df = pd.read_csv(LABEL_DIR / "pfak_combined_label_Annabel_aug2026.csv")
        pfak_fa = pfak_df[pfak_df["label"].isin(LABEL_ORDER_4)][["filename", "label"]]
        pfak_ann = _make_ann_csv(pfak_fa, tmp, "ann_pfak_ctrl.csv")
        print(f"pfak/ctrl FA labels: {len(pfak_fa)} patches")

        # ── ppax/ctrl labels (Ernest, FA subtypes only) ──────────────────────
        ppax_df = pd.read_csv(LABEL_DIR / "ppax_combined_label_Ernest_latest.csv")
        ppax_fa = ppax_df[ppax_df["label"].isin(LABEL_ORDER_4)][["filename", "label"]]
        ppax_ann = _make_ann_csv(ppax_fa, tmp, "ann_ppax_ctrl.csv")
        print(f"ppax/ctrl FA labels: {len(ppax_fa)} patches (Ernest)")

        patch_dirs = [
            # vinc/ctrl: gated to Stage-1 predicted-adhesion patches
            {
                "path":            str(PATCH_DIRS["vinc_ctrl"]),
                "condition":       0,
                "condition_name":  "vinc_ctrl",
                "annotation_file": ctrl_ann,
                "label_col":       "label",
                "filename_col":    "unique_ID",
                "label_order":     LABEL_ORDER_4,
                "val_split":       0.25,
                "patch_include":   adh_fns,
            },
            # vinc/ycomp: all patches; labeled subset gets SupCon loss
            {
                "path":            str(PATCH_DIRS["vinc_ycomp"]),
                "condition":       1,
                "condition_name":  "vinc_ycomp",
                "annotation_file": ycomp_ann,
                "label_col":       "label",
                "filename_col":    "unique_ID",
                "label_order":     LABEL_ORDER_4,
                "val_split":       0.25,
            },
            # pfak/ctrl: all patches; labeled subset gets SupCon loss
            {
                "path":            str(PATCH_DIRS["pfak_ctrl"]),
                "condition":       2,
                "condition_name":  "pfak_ctrl",
                "annotation_file": pfak_ann,
                "label_col":       "label",
                "filename_col":    "unique_ID",
                "label_order":     LABEL_ORDER_4,
                "val_split":       0.25,
            },
            # ppax/ctrl: all patches; Ernest labels get SupCon loss
            {
                "path":            str(PATCH_DIRS["ppax_ctrl"]),
                "condition":       3,
                "condition_name":  "ppax_ctrl",
                "annotation_file": ppax_ann,
                "label_col":       "label",
                "filename_col":    "unique_ID",
                "label_order":     LABEL_ORDER_4,
                "val_split":       0.25,
            },
        ]

        cfg = AEConfig(
            result_dir            = RESULT_DIR,
            patch_dirs            = patch_dirs,
            pretrained_checkpoint = str(PRETRAINED_CKPT),

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

            epochs                  = epochs,
            lr                      = lr,
            batch_size              = batch_size,
            num_workers             = 0,
            val_split               = 0.25,
            group_split             = True,
            weight_decay            = 1e-4,
            warmup_epochs           = 0,
            lr_scheduler            = "none",
            early_stopping_patience = 0,
            min_epochs_for_best     = 0,

            save_recon  = False,
            device      = "auto",
        )

        run_ae_pipeline(cfg)
        print(f"\nStage-2 combined AE done: {RESULT_DIR}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",     type=int,   default=300)
    ap.add_argument("--lr",         type=float, default=5e-4)
    ap.add_argument("--batch-size", type=int,   default=128)
    args = ap.parse_args()
    run(args.epochs, args.lr, args.batch_size)
    print("\nAll done.")


if __name__ == "__main__":
    main()
