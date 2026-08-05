#!/usr/bin/env python3
"""
make_prediction_overlays.py

For each annotated frame (f0000-f0003): 3-panel figure
  Panel A – stitched vinc channel
  Panel B – prediction overlay  (green=adhesion, purple=No adhesion)
  Panel C – annotation vs prediction (green=TP, purple=TN, red=FP, orange=FN)

For unannotated frames: 2-panel figure (raw + prediction only).

Saves PNGs to {result_dir}/fa_cls_zrecon/
  overlay_frame{NNNN}.png          — annotated frames (3-panel)
  overlay_frame{NNNN}_predonly.png — unannotated frames (2-panel)

Usage:
  python scripts/make_prediction_overlays.py [--split s2v2]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import tifffile

DATA_ROOT  = Path("/net/projects/CLS/lding/data/fa_data_analysis")
RUN_DIR    = DATA_ROOT / "ae_results" / "contrastive_run"
PATCH_ROOT = DATA_ROOT / "ae_results" / "patches" / "cio" / "vinc" / "control" / "tiff_patches32_mr10"

PS = 32

# ── Colour palette ────────────────────────────────────────────────────────────
C_AD  = np.array([0.13, 0.63, 0.24, 0.50])   # green  — adhesion
C_NA  = np.array([0.55, 0.18, 0.72, 0.45])   # purple — No adhesion
C_TP  = np.array([0.13, 0.63, 0.24, 0.65])   # green  — TP
C_TN  = np.array([0.55, 0.18, 0.72, 0.60])   # purple — TN
C_FP  = np.array([0.90, 0.10, 0.10, 0.75])   # red    — FP
C_FN  = np.array([1.00, 0.55, 0.00, 0.75])   # orange — FN


def _parse_fname(fn: str):
    m = re.search(r"f(\d+)x(\d+)y(\d+)ps(\d+)", fn)
    return int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))


def _stitch_frame(df_frame: pd.DataFrame):
    xs, ys = df_frame["px"].values, df_frame["py"].values
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    W = (x_max - x_min) + PS
    H = (y_max - y_min) + PS
    canvas = np.zeros((H, W), dtype=np.float32)

    for _, row in df_frame.iterrows():
        fp = PATCH_ROOT / row["filename"]
        if not fp.exists():
            continue
        try:
            img = tifffile.imread(str(fp))
        except Exception:
            continue
        if img.ndim == 3:
            img = img[0]
        xi = row["px"] - x_min
        yi = row["py"] - y_min
        canvas[yi : yi + PS, xi : xi + PS] = img

    lo, hi = np.percentile(canvas[canvas > 0], [1, 99]) if canvas.max() > 0 else (0, 1)
    canvas = np.clip((canvas - lo) / max(hi - lo, 1e-6), 0, 1)
    return canvas, int(x_min), int(y_min)


def _pred_overlay(df_frame: pd.DataFrame, x_min: int, y_min: int, W: int, H: int):
    ov = np.zeros((H, W, 4), dtype=np.float32)
    for _, row in df_frame.iterrows():
        xi, yi = row["px"] - x_min, row["py"] - y_min
        ov[yi : yi + PS, xi : xi + PS] = C_AD if row["pred_label"] == "adhesion" else C_NA
    return ov


def _compare_overlay(df_frame: pd.DataFrame, x_min: int, y_min: int, W: int, H: int):
    ov = np.zeros((H, W, 4), dtype=np.float32)
    for _, row in df_frame.iterrows():
        ann = row.get("annotation_label_name")
        if pd.isna(ann):
            continue
        ann2 = "No adhesion" if ann == "No adhesion" else "adhesion"
        pred = row["pred_label"]
        xi, yi = row["px"] - x_min, row["py"] - y_min
        if   ann2 == "adhesion"    and pred == "adhesion":    color = C_TP
        elif ann2 == "No adhesion" and pred == "No adhesion": color = C_TN
        elif ann2 == "No adhesion" and pred == "adhesion":    color = C_FP
        else:                                                  color = C_FN
        ov[yi : yi + PS, xi : xi + PS] = color
    return ov


def _composite(bg, ov):
    bg_rgb = np.stack([bg, bg, bg], axis=-1)
    a = ov[..., 3:4]
    return np.clip(bg_rgb * (1 - a) + ov[..., :3] * a, 0, 1)


LEG_PRED = [
    mpatches.Patch(color=C_AD[:3], alpha=0.9, label="adhesion (predicted)"),
    mpatches.Patch(color=C_NA[:3], alpha=0.9, label="No adhesion (predicted)"),
]
LEG_CMP = [
    mpatches.Patch(color=C_TP[:3], label="TP  (both adhesion)"),
    mpatches.Patch(color=C_TN[:3], label="TN  (both no-adh)"),
    mpatches.Patch(color=C_FP[:3], label="FP  (pred=adh, ann=no-adh)"),
    mpatches.Patch(color=C_FN[:3], label="FN  (pred=no-adh, ann=adh)"),
]


def make_annotated_overlay(df: pd.DataFrame, frame: int, out_path: Path):
    """3-panel: raw | prediction | annotation vs prediction."""
    df_f = df[df["frame"] == frame].copy()
    bg, x_min, y_min = _stitch_frame(df_f)
    H, W = bg.shape

    comp_pred = _composite(bg, _pred_overlay(df_f, x_min, y_min, W, H))
    comp_cmp  = _composite(bg, _compare_overlay(df_f, x_min, y_min, W, H))

    n_ann    = df_f["annotation_label_name"].notna().sum()
    n_ad     = (df_f["pred_label"] == "adhesion").sum()
    n_na     = (df_f["pred_label"] == "No adhesion").sum()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), facecolor="white",
                             gridspec_kw={"wspace": 0.04})
    for ax, img, title in zip(axes,
            [bg, comp_pred, comp_cmp],
            [f"Vinc channel  (frame f{frame:04d})",
             f"Prediction  —  adhesion={n_ad}  /  no-adh={n_na}",
             f"Annotation vs Prediction  ({n_ann} labelled patches)"]):
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None,
                  origin="upper", interpolation="nearest")
        ax.set_title(title, fontsize=11, pad=4)
        ax.axis("off")

    axes[1].legend(handles=LEG_PRED, loc="lower right", fontsize=8, framealpha=0.85)
    axes[2].legend(handles=LEG_CMP,  loc="lower right", fontsize=8, framealpha=0.85)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def make_predonly_overlay(df: pd.DataFrame, frame: int, out_path: Path):
    """2-panel: raw | prediction (no annotation panel)."""
    df_f = df[df["frame"] == frame].copy()
    bg, x_min, y_min = _stitch_frame(df_f)
    H, W = bg.shape
    comp_pred = _composite(bg, _pred_overlay(df_f, x_min, y_min, W, H))

    n_ad = (df_f["pred_label"] == "adhesion").sum()
    n_na = (df_f["pred_label"] == "No adhesion").sum()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5), facecolor="white",
                             gridspec_kw={"wspace": 0.04})
    for ax, img, title in zip(axes,
            [bg, comp_pred],
            [f"Vinc channel  (frame f{frame:04d})",
             f"Prediction  —  adhesion={n_ad}  /  no-adh={n_na}"]):
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None,
                  origin="upper", interpolation="nearest")
        ax.set_title(title, fontsize=11, pad=4)
        ax.axis("off")

    axes[1].legend(handles=LEG_PRED, loc="lower right", fontsize=8, framealpha=0.85)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


SPLITS = ["s1v3", "s2v2", "s3v1"]
SPLIT_LABELS = {
    "s1v3": "s1v3  (66 adh train patches)",
    "s2v2": "s2v2  (99 adh train patches)",
    "s3v1": "s3v1  (151 adh train patches)",
}


def make_split_comparison(frame: int, out_dir: Path):
    """4-panel: raw | s1v3 pred | s2v2 pred | s3v1 pred — for one frame."""
    # Load all three splits' predictions for this frame
    dfs = {}
    for sp in SPLITS:
        p = RUN_DIR / f"annabel_vinc_supcon2_{sp}" / "fa_cls_zrecon" / "predictions_all.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df[["frame", "px", "py", "ps"]] = df["filename"].apply(
            lambda f: pd.Series(_parse_fname(f)))
        dfs[sp] = df[df["frame"] == frame].copy()

    if not dfs:
        return

    # Use s2v2 frame as the reference background (all splits have same patches)
    ref = next(iter(dfs.values()))
    bg, x_min, y_min = _stitch_frame(ref)
    H, W = bg.shape

    panels = [bg]
    titles = [f"Vinc channel  (frame f{frame:04d})"]
    for sp in SPLITS:
        if sp not in dfs:
            continue
        ov = _pred_overlay(dfs[sp], x_min, y_min, W, H)
        n_ad = (dfs[sp]["pred_label"] == "adhesion").sum()
        n_na = (dfs[sp]["pred_label"] == "No adhesion").sum()
        panels.append(_composite(bg, ov))
        titles.append(f"{SPLIT_LABELS[sp]}\nadh={n_ad}  no-adh={n_na}")

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 6.5),
                             facecolor="white", gridspec_kw={"wspace": 0.04})
    for ax, img, title in zip(axes, panels, titles):
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None,
                  origin="upper", interpolation="nearest")
        ax.set_title(title, fontsize=10, pad=4)
        ax.axis("off")
    axes[-1].legend(handles=LEG_PRED, loc="lower right", fontsize=8, framealpha=0.85)

    fig.tight_layout()
    out_path = out_dir / f"split_comparison_frame{frame:04d}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="s2v2")
    ap.add_argument("--n-extra", type=int, default=3,
                    help="Number of unannotated frames to include")
    ap.add_argument("--compare-splits", action="store_true",
                    help="Also generate split-comparison figures (all 3 splits on same frame)")
    args = ap.parse_args()

    result_dir = RUN_DIR / f"annabel_vinc_supcon2_{args.split}"
    pred_csv   = result_dir / "fa_cls_zrecon" / "predictions_all.csv"
    out_dir    = result_dir / "fa_cls_zrecon"

    print(f"Loading {pred_csv.name} ...")
    df = pd.read_csv(pred_csv)
    df[["frame", "px", "py", "ps"]] = df["filename"].apply(
        lambda f: pd.Series(_parse_fname(f)))

    ann_frames = sorted(df[df["annotation_label_name"].notna()]["frame"].unique())
    all_frames = sorted(df["frame"].unique())
    extra_frames = [f for f in all_frames if f not in ann_frames][: args.n_extra]

    print(f"Annotated frames: {ann_frames}")
    for fr in ann_frames:
        make_annotated_overlay(df, fr, out_dir / f"overlay_frame{fr:04d}.png")

    print(f"Unannotated frames: {extra_frames}")
    for fr in extra_frames:
        make_predonly_overlay(df, fr, out_dir / f"overlay_frame{fr:04d}_predonly.png")

    if args.compare_splits:
        print("Generating split-comparison figures ...")
        comp_out = RUN_DIR / "split_comparison_overlays"
        for fr in ann_frames + extra_frames:
            make_split_comparison(fr, comp_out)

    print("Done.")


if __name__ == "__main__":
    main()
