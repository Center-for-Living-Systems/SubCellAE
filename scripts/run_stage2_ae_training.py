#!/usr/bin/env python3
"""
run_stage2_ae_training.py

Train a dedicated Stage-2 SupCon AE for 4-class FA subtype discrimination.

Design:
  - Stage 1 (binary 2-class AE+LightGBM) is already trained and fixed.
  - Stage 2 trains a NEW SupCon AE using Annabel's 4-class adhesion labels
    (Nascent Adhesion / focal complex / focal adhesion / fibrillar adhesion).
  - Same vinc/control patch set (all 14879 patches for reconstruction).
  - Per-image group_split — same frames as Stage 1:
      s1v3: frame 0 → train SupCon,  frames 1,2,3 → val
      s2v2: frames 0,1 → train SupCon, frames 2,3 → val
      s3v1: frames 0,1,2 → train SupCon, frame 3 → val
  - No-adhesion patches are treated as unlabeled (label_order has only 4 types).
  - After AE training, trains Stage-2 LightGBM on train-split adhesion latents.
  - Evaluates on val-split adhesion patches.

Output:
  {contrastive_run}/annabel_vinc_supcon2_stage2_{split}/
    model_best.pt
    latents.csv
    stage2_cls/
      model.pkl
      metrics.csv
      confusion_matrix_norm.png

Usage:
  python scripts/run_stage2_ae_training.py [--split s2v2] [--epochs 300]
  python scripts/run_stage2_ae_training.py --all-splits
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DATA_ROOT  = Path("/net/projects/CLS/lding/data/fa_data_analysis")
LABEL_DIR  = DATA_ROOT / "labelling"
RUN_DIR    = DATA_ROOT / "ae_results" / "contrastive_run"
PATCH_BASE = DATA_ROOT / "ae_results" / "patches"
PATCH_DIR  = PATCH_BASE / "cio/vinc/control/tiff_patches32_mr10"

LABEL_ORDER_4 = [
    "Nascent Adhesion",
    "focal complex",
    "focal adhesion",
    "fibrillar adhesion",
]

# Stage 1 split used to gate which patches enter Stage 2 training
STAGE1_GATE_SPLIT = "s2v2"

# val_split config per split name (fraction assigned to val)
# group_split=True → splits by image group; val_split is approximate
SPLIT_VAL_FRAC = {
    "s1v3": 0.75,   # 1 train image / 3 val images
    "s2v2": 0.50,   # 2 train images / 2 val images
    "s3v1": 0.25,   # 3 train images / 1 val image
}


def _get_stage1_adhesion_filenames() -> list[str]:
    """Run Stage-1 (s2v2 gate) on all vinc/control patches; return filenames
    of patches predicted as adhesion."""
    import joblib
    import numpy as np
    import pandas as pd

    stage1_dir = RUN_DIR / f"annabel_vinc_supcon2_{STAGE1_GATE_SPLIT}"
    lat = pd.read_csv(stage1_dir / "blind_test" / "vinc_control_latents.csv")
    z_cols = [c for c in lat.columns if c.startswith("z_")]
    stage1 = joblib.load(str(stage1_dir / "fa_cls_zrecon" / "model.pkl"))
    preds = stage1.predict(lat[z_cols].values)
    adh_fns = lat.loc[preds == 1, "filename"].tolist()
    print(f"Stage-1 gate ({STAGE1_GATE_SPLIT}): {len(adh_fns)} / {len(lat)} patches predicted as adhesion")
    return adh_fns


def run_split(split: str, epochs: int, lr: float, batch_size: int,
              adh_filenames: list[str]):
    from subcellae.pipeline.ae_pipeline import AEConfig, run_ae_pipeline

    val_frac = SPLIT_VAL_FRAC[split]
    stage1_dir = RUN_DIR / f"annabel_vinc_supcon2_{split}"
    result_dir = RUN_DIR / f"annabel_vinc_supcon2_stage2_{split}"

    if not stage1_dir.exists():
        raise FileNotFoundError(f"Stage-1 dir not found: {stage1_dir}")

    print(f"\n{'='*65}")
    print(f"Stage-2 AE training  split={split}  val_frac={val_frac}")
    print(f"Input: {len(adh_filenames)} predicted-adhesion patches (Stage-1 gate: {STAGE1_GATE_SPLIT})")
    print(f"Output: {result_dir}")
    print(f"{'='*65}")

    cfg = AEConfig(
        result_dir=result_dir,

        patch_dirs=[{
            "path":            str(PATCH_DIR),
            "condition":       0,
            "condition_name":  "vinc_control",
            "annotation_file": str(LABEL_DIR / "vinc_control_label_Annabel_20260715_1554.csv"),
            "label_col":       "label",
            "filename_col":    "unique_ID",     # hyphen format matches _patch_name_to_annotation_key
            "label_order":     LABEL_ORDER_4,   # no-adh patches unlabeled (-1), only 4 adh types
            "patch_include":   adh_filenames,   # ONLY predicted-adhesion patches enter AE
        }],

        # Same architecture as Stage 1
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
    print(f"\nStage-2 AE done: {result_dir}")

    # ── Train Stage-2 LightGBM on train-split adhesion latents ────────────
    _train_stage2_cls(result_dir, split)


def _train_stage2_cls(result_dir: Path, split: str):
    """Train 4-class LightGBM on train-split adhesion patches; eval on val-split.

    The patch dataset has 50 frames; Annabel labels cover only frames 0-3.
    The AE group_split randomly assigns frames to train/val across all 50 frames,
    which can accidentally put ALL labeled frames into val.  To avoid this we use
    explicit frame-based train/val assignment for the LightGBM (same as Stage 1):
      s1v3: frame 0 → train,  frames 1-3 → val
      s2v2: frames 0-1 → train,  frames 2-3 → val
      s3v1: frames 0-2 → train,  frame 3 → val
    """
    import numpy as np
    import pandas as pd
    import joblib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        ConfusionMatrixDisplay, balanced_accuracy_score,
    )

    try:
        from lightgbm import LGBMClassifier
        CLF_CLASS = LGBMClassifier
        clf_kwargs = dict(
            n_estimators=500, learning_rate=0.05, num_leaves=31,
            min_child_samples=3, class_weight="balanced",
            random_state=42, verbose=-1, n_jobs=1,
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.utils.class_weight import compute_sample_weight
        CLF_CLASS = None
        clf_kwargs = {}

    lat = pd.read_csv(result_dir / "latents.csv")
    z_cols = [c for c in lat.columns if c.startswith("z_")]
    label_col = "annotation_label_name" if "annotation_label_name" in lat.columns else "label"
    ADHESION_TYPES = set(LABEL_ORDER_4)

    # Fixed frame-based split for the 4-class LightGBM (independent of AE group_split)
    LABEL_TRAIN_FRAMES = {
        "s1v3": {"vinc_control_f0000"},
        "s2v2": {"vinc_control_f0000", "vinc_control_f0001"},
        "s3v1": {"vinc_control_f0000", "vinc_control_f0001", "vinc_control_f0002"},
    }
    LABEL_VAL_FRAMES = {
        "s1v3": {"vinc_control_f0001", "vinc_control_f0002", "vinc_control_f0003"},
        "s2v2": {"vinc_control_f0002", "vinc_control_f0003"},
        "s3v1": {"vinc_control_f0003"},
    }
    train_frames = LABEL_TRAIN_FRAMES[split]
    val_frames   = LABEL_VAL_FRAMES[split]

    all_labeled = lat[lat[label_col].isin(ADHESION_TYPES)]
    train_ad = all_labeled[all_labeled["group"].isin(train_frames)]
    val_ad   = all_labeled[all_labeled["group"].isin(val_frames)]

    print(f"\nStage-2 LightGBM  split={split}  (fixed label-frame split)")
    print(f"  Train frames: {sorted(train_frames)}")
    print(f"  Train: {len(train_ad)} adhesion patches  {train_ad[label_col].value_counts().to_dict()}")
    print(f"  Val frames: {sorted(val_frames)}")
    print(f"  Val:   {len(val_ad)} adhesion patches  {val_ad[label_col].value_counts().to_dict()}")

    if len(train_ad) == 0:
        print(f"  WARNING: 0 labeled adhesion patches in train split — skipping.")
        return

    lo4_present = [c for c in LABEL_ORDER_4 if c in set(train_ad[label_col])]
    lo4_to_int  = {c: i for i, c in enumerate(lo4_present)}

    X_tr = train_ad[z_cols].values.astype(np.float32)
    y_tr = np.array([lo4_to_int[l] for l in train_ad[label_col]])

    if CLF_CLASS is not None:
        clf = CLF_CLASS(**clf_kwargs)
        clf.fit(X_tr, y_tr)
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.utils.class_weight import compute_sample_weight
        w = compute_sample_weight("balanced", y_tr)
        clf = GradientBoostingClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )
        clf.fit(X_tr, y_tr, sample_weight=w)

    out_dir = result_dir / "stage2_cls"
    out_dir.mkdir(exist_ok=True)
    joblib.dump(clf, str(out_dir / "model.pkl"))
    print(f"  Saved: {out_dir}/model.pkl")

    if len(val_ad) == 0:
        print("  No val adhesion patches — skipping eval")
        return

    X_val = val_ad[z_cols].values.astype(np.float32)
    y_val_str = val_ad[label_col].tolist()
    y_pred_int = clf.predict(X_val)
    y_pred_str = [lo4_present[int(p)] if int(p) < len(lo4_present) else "unknown"
                  for p in y_pred_int]

    acc  = sum(a == b for a, b in zip(y_val_str, y_pred_str)) / len(y_val_str)
    present = [c for c in LABEL_ORDER_4 if c in set(y_val_str)]
    print(f"\n  Val accuracy: {acc*100:.1f}%")
    try:
        bal = balanced_accuracy_score(y_val_str, y_pred_str)
        print(f"  Val balanced acc: {bal*100:.1f}%")
    except Exception:
        bal = float("nan")
    print(classification_report(y_val_str, y_pred_str, labels=present, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_val_str, y_pred_str, labels=present)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="white")
    disp = ConfusionMatrixDisplay(cm_norm, display_labels=[c[:10] for c in present])
    disp.plot(ax=ax, colorbar=True, values_format=".2f")
    ax.set_title(f"Stage-2 4-class  {split}  val  (row-normalised)\nacc={acc*100:.1f}%  bal={bal*100:.1f}%", fontsize=9)
    fig.tight_layout()
    fig.savefig(str(out_dir / "confusion_matrix_norm.png"), dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame([{
        "split": split, "n_train": len(train_ad), "n_val": len(val_ad),
        "accuracy": acc, "balanced_acc": bal,
        "label_order": str(lo4_present),
    }]).to_csv(str(out_dir / "metrics.csv"), index=False)
    print(f"  Saved: {out_dir}/metrics.csv  {out_dir}/confusion_matrix_norm.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split",  default="s2v2", choices=["s1v3", "s2v2", "s3v1"])
    ap.add_argument("--all-splits", action="store_true")
    ap.add_argument("--epochs", type=int,   default=300)
    ap.add_argument("--lr",     type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    # Get predicted-adhesion filenames from Stage 1 gate once, shared across splits
    adh_filenames = _get_stage1_adhesion_filenames()

    splits = ["s1v3", "s2v2", "s3v1"] if args.all_splits else [args.split]
    for sp in splits:
        run_split(sp, args.epochs, args.lr, args.batch_size, adh_filenames)

    print("\nAll done.")


if __name__ == "__main__":
    main()
