#!/usr/bin/env python3
"""
run_image_efficiency.py
=======================
Evaluate trained image-efficiency AE runs.

For each ie_n{N:03d}_r{repeat} directory:
  1. Load latents.csv  (encoded training-frame patches)
  2. Train LGBM on frame-0 labels  (npi = all)
  3. Encode test-frame patches (frames 1, 2, 3) using the saved model
  4. Evaluate: balanced accuracy + per-class recall & precision

Outputs
-------
  results/img_eff_results.csv
  results/img_eff_curve.png
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import torch
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import GradientBoostingClassifier

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from subcellae.modelling.autoencoders import ContrastiveAE

# ---------------------------------------------------------------------------
DATA      = Path("/net/projects/CLS/lding/data/fa_data_analysis")
IE_DIR    = DATA / "ae_results/contrastive_run/img_eff"
PATCH_DIR = DATA / "ae_results/patches/cio/vinc/control/tiff_patches32_mr10"
LAB_DIR   = DATA / "labelling"
FULL_ANN  = LAB_DIR / "vinc_control_label_Annabel_20260715_1554_2cls.csv"

Z_COLS  = [f"z_{i}" for i in range(12)]
N_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 47]

TEST_FRAMES  = {1, 2, 3}
TRAIN_LABEL_FRAME = 0

# Model hyperparams (must match the training YAML)
LATENT_DIM    = 12
PROJ_DIM      = 8
INPUT_PS      = 32
INPUT_DIVISOR = 2.0

# ---------------------------------------------------------------------------

def _extract_frame(fn: str) -> int:
    m = re.search(r"_f(\d+)", fn)
    return int(m.group(1)) if m else -1


def _load_model(run_dir: Path) -> ContrastiveAE:
    ckpt = run_dir / "model_final.pt"
    obj = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        model = ContrastiveAE(latent_dim=LATENT_DIM, proj_dim=PROJ_DIM,
                              input_ps=INPUT_PS, no_ch=1, BN_flag=False,
                              output_sigmoid=False)
        model.load_state_dict(obj)
    else:
        model = obj
    model.eval()
    return model


def _encode_frames(model: ContrastiveAE, frame_set: set[int],
                   batch_size: int = 256) -> pd.DataFrame:
    """Load patches from PATCH_DIR for the given frames and encode them."""
    patch_re = re.compile(r"^(.+)_f(\d{4})x\d+y\d+ps\d+\.tiff?$", re.IGNORECASE)
    patch_files = []
    for fname in sorted(PATCH_DIR.iterdir()):
        m = patch_re.match(fname.name)
        if not m:
            continue
        fidx = int(m.group(2))
        if fidx in frame_set:
            patch_files.append(fname)

    if not patch_files:
        return pd.DataFrame()

    records = []
    for i in range(0, len(patch_files), batch_size):
        batch_paths = patch_files[i : i + batch_size]
        imgs = []
        for p in batch_paths:
            img = tifffile.imread(str(p)).astype(np.float32) / INPUT_DIVISOR
            imgs.append(img)
        x = torch.from_numpy(np.stack(imgs)).unsqueeze(1)   # (B, 1, 32, 32)
        with torch.no_grad():
            z = model.encode(x).numpy()                     # (B, 12)
        for p, z_vec in zip(batch_paths, z):
            row = {"filename": p.name}
            for j, v in enumerate(z_vec):
                row[f"z_{j}"] = float(v)
            records.append(row)

    return pd.DataFrame(records)


def run_one(run_dir: Path, full_ann: pd.DataFrame) -> dict | None:
    m = re.match(r"ie_n(\d+)_r(\d+)$", run_dir.name)
    if not m:
        return None
    n_images = int(m.group(1))
    repeat   = int(m.group(2))

    lat_csv = run_dir / "latents.csv"
    ckpt    = run_dir / "model_final.pt"
    if not lat_csv.exists() or not ckpt.exists():
        print(f"  [skip] not trained yet: {run_dir.name}")
        return None

    # ── train LGBM on frame-0 labels from latents.csv ───────────────────────
    latents = pd.read_csv(lat_csv)
    latents["frame"] = latents["filename"].apply(_extract_frame)

    train_latents = latents[latents["frame"] == TRAIN_LABEL_FRAME].copy()
    train_labeled = train_latents.merge(
        full_ann[["filename", "label"]], on="filename", how="inner"
    )
    if len(train_labeled) == 0:
        print(f"  [skip] no frame-0 labels in latents: {run_dir.name}")
        return None

    le = LabelEncoder()
    y_train = le.fit_transform(train_labeled["label"])
    X_train = train_labeled[Z_COLS].values
    w_train = compute_sample_weight("balanced", y_train)

    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
    )
    clf.fit(X_train, y_train, sample_weight=w_train)

    # ── encode test frames using the trained AE ──────────────────────────────
    model     = _load_model(run_dir)
    test_enc  = _encode_frames(model, TEST_FRAMES)
    if test_enc.empty:
        print(f"  [skip] no test patches found: {run_dir.name}")
        return None

    test_enc["frame"] = test_enc["filename"].apply(_extract_frame)
    test_labeled = test_enc.merge(full_ann[["filename", "label"]], on="filename", how="inner")
    if len(test_labeled) == 0:
        print(f"  [skip] no test labels: {run_dir.name}")
        return None

    y_test = le.transform(test_labeled["label"])
    X_test = test_labeled[Z_COLS].values
    y_pred = clf.predict(X_test)

    bacc       = balanced_accuracy_score(y_test, y_pred)
    recalls    = recall_score(   y_test, y_pred, labels=[0, 1], average=None, zero_division=0)
    precisions = precision_score(y_test, y_pred, labels=[0, 1], average=None, zero_division=0)

    result = {
        "run":            run_dir.name,
        "n_images":       n_images,
        "repeat":         repeat,
        "n_train_labels": len(train_labeled),
        "n_test":         len(test_labeled),
        "balanced_acc":   float(bacc),
        "adh_recall":     float(recalls[1]),
        "noad_recall":    float(recalls[0]),
        "adh_precision":  float(precisions[1]),
        "noad_precision": float(precisions[0]),
    }
    print(f"  {run_dir.name}  n_img={n_images}  BAcc={bacc:.3f}  "
          f"adh_rec={recalls[1]:.3f}  noad_rec={recalls[0]:.3f}")
    return result


def main():
    full_ann = pd.read_csv(FULL_ANN)
    full_ann["frame"] = full_ann["filename"].apply(_extract_frame)

    run_dirs = sorted(IE_DIR.glob("ie_n*_r*"))
    if not run_dirs:
        print(f"No run directories found in {IE_DIR}")
        return

    print(f"Found {len(run_dirs)} run directories")
    records = []
    for rd in run_dirs:
        r = run_one(rd, full_ann)
        if r:
            records.append(r)

    if not records:
        print("No completed runs to evaluate.")
        return

    df = pd.DataFrame(records)
    out_csv = _REPO / "results" / "img_eff_results.csv"
    out_csv.parent.mkdir(exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # ── summary ──────────────────────────────────────────────────────────────
    agg_cols = ["balanced_acc", "adh_recall", "noad_recall",
                "adh_precision", "noad_precision"]
    summary = (df.groupby("n_images")[agg_cols]
                 .agg(["mean", "std"])
                 .reset_index())
    summary.columns = ["n_images"] + [
        f"{m}_{s}" for m in agg_cols for s in ("mean", "std")
    ]
    summary = summary.sort_values("n_images")
    print("\n" + summary[["n_images", "balanced_acc_mean", "balanced_acc_std"]].to_string(index=False))

    # ── plot ─────────────────────────────────────────────────────────────────
    CURVES = [
        ("balanced_acc", "Balanced accuracy",   "o-",  "#333333", 2.0),
        ("adh_recall",   "Adhesion recall",     "s--", "#2ca02c", 1.8),
        ("noad_recall",  "No-adhesion recall",  "^:",  "#1f77b4", 1.8),
    ]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
    x = np.log2(summary["n_images"].values)   # log scale for x

    for col, label, fmt, color, lw in CURVES:
        means = summary[f"{col}_mean"].values * 100
        stds  = summary[f"{col}_std"].values  * 100
        stds  = np.nan_to_num(stds, nan=0.0)
        ax.errorbar(x, means, yerr=stds, fmt=fmt, color=color,
                    capsize=3, linewidth=lw, markersize=6, label=label)
        for xi, m in zip(x, means):
            ax.text(xi, m + 1.5, f"{m:.0f}%", ha="center", fontsize=7, color=color)

    n_vals = summary["n_images"].values
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in n_vals], fontsize=9)
    ax.set_xlabel("Number of AE training images (N)", fontsize=11)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_ylim(40, 112)
    ax.axhline(90, color="#CCCCCC", linestyle="--", linewidth=0.8, zorder=0)
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=10, loc="lower right")
    ax.set_title(
        "Image-count Efficiency — SupCon AE + LGBM (vinc / control)\n"
        "Classifier trained on frame-0 labels (npi=all)  ·  Test: frames 1, 2, 3\n"
        "AE sees N images: frame 0 (labeled) + N−1 unlabeled frames 4–49",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout()
    out_fig = _REPO / "results" / "img_eff_curve.png"
    fig.savefig(str(out_fig), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_fig}")


if __name__ == "__main__":
    main()
