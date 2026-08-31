#!/usr/bin/env python3
"""
analyze_noad_ad_errors.py
=========================
Error analysis for the binary adhesion / no-adhesion classifier.

Supports all three frame-split configs (--cfg 0/1/2):
  cfg 0 = s1v3: train frames [0],    test frames [1,2,3]
  cfg 1 = s2v2: train frames [0,1],  test frames [2,3]
  cfg 2 = s3v1: train frames [0,1,2], test frames [3]

Uses npi=100, repeat 0 for each cfg.  Identifies FN/FP errors and attaches
4-class FA sub-label + 99th-pct intensity to each patch.

Outputs (in results/)
---------------------
  noad_ad_errors_grid_cfg{c}.png      — error patch grid; border = FA sub-label colour
  noad_ad_errors_intensity_cfg{c}.png — intensity of errors vs correct by FA sub-label
  noad_ad_error_predictions_cfg{c}.csv
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from lightgbm import LGBMClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

# ---------------------------------------------------------------------------
DATA   = Path("/net/projects/CLS/lding/data/fa_data_analysis")
LAB    = DATA / "labelling"
PATCH  = DATA / "ae_results/patches/cio/vinc/control/tiff_patches32_mr10"
LE_DIR = DATA / "ae_results/contrastive_run/le_clean"

FULL_2CLS = LAB / "vinc_control_label_Annabel_20260715_1554_2cls.csv"
FULL_4CLS = LAB / "vinc_control_label_Annabel_20260715_1554.csv"

CONFIGS = {
    0: dict(label="s1v3", train_frames={0},       test_frames={1, 2, 3}),
    1: dict(label="s2v2", train_frames={0, 1},    test_frames={2, 3}),
    2: dict(label="s3v1", train_frames={0, 1, 2}, test_frames={3}),
}
Z_COLS      = [f"z_{i}" for i in range(12)]

# 5-class colour scheme (same as other illustrations)
FA5_COLORS = {
    "No adhesion":        "#9467bd",   # purple
    "Nascent Adhesion":   "#1f77b4",   # blue
    "focal complex":      "#ff7f0e",   # orange
    "focal adhesion":     "#2ca02c",   # green
    "fibrillar adhesion": "#d62728",   # red
}
FA5_SHORT = {
    "No adhesion":        "NoAd",
    "Nascent Adhesion":   "NA",
    "focal complex":      "FC",
    "focal adhesion":     "FA",
    "fibrillar adhesion": "Fib",
}
FA5_ORDER = ["No adhesion", "Nascent Adhesion", "focal complex",
             "focal adhesion", "fibrillar adhesion"]

BORDER_PX  = 3   # thickness of coloured bounding box in pixels
PATCH_SIZE = 32
INTENSITY_PERCENTILE = 99

OUT_DIR = _REPO / "results"


# ---------------------------------------------------------------------------

def _frame(fn: str) -> int:
    m = re.search(r"f(\d+)", fn)
    return int(m.group(1)) if m else -1


def _peak(img: np.ndarray) -> float:
    return float(np.percentile(img.astype(np.float32), INTENSITY_PERCENTILE))


def _norm(img: np.ndarray) -> np.ndarray:
    lo, hi = img.min(), img.max()
    return (img - lo) / (hi - lo + 1e-9)


def load_patch(fname: str) -> np.ndarray | None:
    p = PATCH / fname
    if not p.exists():
        return None
    return tifffile.imread(str(p)).astype(np.float32)


def add_border(img_norm: np.ndarray, color_hex: str, px: int = BORDER_PX) -> np.ndarray:
    """Return RGB image (H+2px, W+2px, 3) with coloured border."""
    r = int(color_hex[1:3], 16) / 255
    g = int(color_hex[3:5], 16) / 255
    b = int(color_hex[5:7], 16) / 255
    h, w = img_norm.shape
    rgb = np.stack([img_norm, img_norm, img_norm], axis=-1)  # (H, W, 3)
    out = np.ones((h + 2 * px, w + 2 * px, 3), dtype=np.float32)
    out[:, :, 0] = r
    out[:, :, 1] = g
    out[:, :, 2] = b
    out[px:px + h, px:px + w] = rgb
    return out


# ---------------------------------------------------------------------------

def build_predictions(cfg_id: int) -> tuple[pd.DataFrame, float]:
    cfg          = CONFIGS[cfg_id]
    train_frames = cfg["train_frames"]
    test_frames  = cfg["test_frames"]
    run_dir      = LE_DIR / f"le_c{cfg_id}_npi100_r0"
    ann_csv      = LAB / "le_clean" / f"le_c{cfg_id}_npi100_r0.csv"

    lat = pd.read_csv(run_dir / "latents.csv")
    lat["frame"] = lat["filename"].apply(_frame)

    ann_train = pd.read_csv(ann_csv)
    train_lat = lat[lat["frame"].isin(train_frames)].merge(
        ann_train[["filename", "label"]], on="filename", how="inner"
    )

    full2    = pd.read_csv(FULL_2CLS)
    test_lat = lat[lat["frame"].isin(test_frames)].merge(
        full2[["filename", "label"]], on="filename", how="inner"
    )

    le   = LabelEncoder()
    y_tr = le.fit_transform(train_lat["label"])
    X_tr = train_lat[Z_COLS].values
    w_tr = compute_sample_weight("balanced", y_tr)

    clf = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        min_child_samples=3, class_weight="balanced",
        random_state=42, verbose=-1, n_jobs=1,
    )
    clf.fit(X_tr, y_tr, sample_weight=w_tr)

    X_te = test_lat[Z_COLS].values
    y_te = le.transform(test_lat["label"])
    y_pr = clf.predict(X_te)

    bacc = balanced_accuracy_score(y_te, y_pr)
    print(f"  BAcc={bacc:.3f}  train={len(train_lat)}  test={len(test_lat)}")

    pred_df = test_lat[["filename", "frame", "label"]].copy()
    pred_df = pred_df.rename(columns={"label": "true_2cls"})
    pred_df["pred_2cls"] = le.inverse_transform(y_pr)
    pred_df["correct"]   = pred_df["true_2cls"] == pred_df["pred_2cls"]
    return pred_df, bacc


def attach_fa4(pred_df: pd.DataFrame) -> pd.DataFrame:
    full4 = pd.read_csv(FULL_4CLS)[["filename", "label"]].rename(
        columns={"label": "fa_label"}
    )
    merged = pred_df.merge(full4, on="filename", how="left")
    # Patches labelled 'No adhesion' in 4cls keep that label
    merged["fa_label"] = merged["fa_label"].fillna("No adhesion")
    return merged


def attach_intensity(df: pd.DataFrame) -> pd.DataFrame:
    intensities = []
    for fn in df["filename"]:
        img = load_patch(fn)
        intensities.append(_peak(img) if img is not None else float("nan"))
    df = df.copy()
    df["peak_intensity"] = intensities
    return df


# ---------------------------------------------------------------------------
# Plots

def plot_error_grid(df: pd.DataFrame, out: Path, cfg_label: str = ""):
    """Two sections: FN errors (adhesion→no-ad) and FP errors (no-ad→adhesion).
    Each patch shown with thick coloured border = FA sub-label colour."""
    fn_df = df[(df["true_2cls"] == "adhesion")    & (~df["correct"])].copy()
    fp_df = df[(df["true_2cls"] == "No adhesion") & (~df["correct"])].copy()

    # Sort FN by FA sub-label order
    fn_df["fa_order"] = fn_df["fa_label"].map(
        {c: i for i, c in enumerate(FA5_ORDER)}
    ).fillna(99)
    fn_df = fn_df.sort_values("fa_order")

    print(f"  FN errors: {len(fn_df)}  (adhesion → predicted no-adhesion)")
    print(f"  FP errors: {len(fp_df)}  (no-adhesion → predicted adhesion)")
    if len(fn_df):
        print("  FN FA sub-labels:", fn_df["fa_label"].value_counts().to_dict())
    if len(fp_df):
        print("  FP FA sub-labels:", fp_df["fa_label"].value_counts().to_dict())

    def render_row(sub_df: pd.DataFrame, max_per_row: int = 20) -> list[np.ndarray]:
        imgs = []
        for _, row in sub_df.head(max_per_row).iterrows():
            img = load_patch(row["filename"])
            if img is None:
                continue
            color = FA5_COLORS.get(row["fa_label"], "#cccccc")
            imgs.append(add_border(_norm(img), color))
        return imgs

    fn_imgs = render_row(fn_df)
    fp_imgs = render_row(fp_df)

    cell_h = PATCH_SIZE + 2 * BORDER_PX
    cell_w = PATCH_SIZE + 2 * BORDER_PX

    max_n = max(len(fn_imgs), len(fp_imgs), 1)
    fig_w = min(max_n * (cell_w / 50), 28)
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, 4.5),
                             gridspec_kw={"hspace": 0.55})

    for ax, imgs, title in zip(
        axes,
        [fn_imgs, fp_imgs],
        [f"FN — adhesion predicted as No-adhesion  (n={len(fn_df)})",
         f"FP — No-adhesion predicted as adhesion  (n={len(fp_df)})"],
    ):
        ax.set_title(title, fontsize=9, fontweight="bold", loc="left")
        ax.axis("off")
        if not imgs:
            ax.text(0.5, 0.5, "no errors", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="gray")
            continue

        strip = np.concatenate(imgs, axis=1)
        ax.imshow(strip, aspect="equal", interpolation="nearest")

        # Draw thin white dividers between patches
        for i in range(1, len(imgs)):
            ax.axvline(i * cell_w - 0.5, color="white", linewidth=0.5, alpha=0.5)

        # Per-patch label below
        for i, (_, row) in enumerate(
            (fn_df if "FN" in title else fp_df).head(len(imgs)).iterrows()
        ):
            short = FA5_SHORT.get(row["fa_label"], "?")
            ax.text(
                i * cell_w + cell_w / 2, cell_h + 1,
                short, ha="center", va="top", fontsize=5.5,
                color=FA5_COLORS.get(row["fa_label"], "#333"),
                fontweight="bold",
            )

    # Legend
    handles = [
        mpatches.Patch(color=FA5_COLORS[c], label=f"{FA5_SHORT[c]} = {c}")
        for c in FA5_ORDER
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f"Ad / No-ad classifier — misclassified patches  [{cfg_label}]\n"
        "Border colour = FA sub-label (5-class)",
        fontsize=10, fontweight="bold",
    )
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_intensity(df: pd.DataFrame, out: Path, cfg_label: str = ""):
    """Strip plot of 99th-pct intensity by FA sub-label; errors highlighted."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    for ax, error_type, title in zip(
        axes,
        ["FN", "FP"],
        ["FN errors: adhesion → predicted No-adhesion",
         "FP errors: No-adhesion → predicted adhesion"],
    ):
        if error_type == "FN":
            err_df  = df[(df["true_2cls"] == "adhesion")    & (~df["correct"])]
            corr_df = df[(df["true_2cls"] == "adhesion")    &   df["correct"]]
        else:
            err_df  = df[(df["true_2cls"] == "No adhesion") & (~df["correct"])]
            corr_df = df[(df["true_2cls"] == "No adhesion") &   df["correct"]]

        rng = np.random.default_rng(42)

        def jitter(n: int, spread: float = 0.15) -> np.ndarray:
            return rng.uniform(-spread, spread, n)

        # Always show all 5 FA sub-classes (even if no data for that class)
        classes_present = FA5_ORDER

        for xi, cls in enumerate(classes_present):
            color = FA5_COLORS[cls]

            # Correctly classified
            sub_c = corr_df[corr_df["fa_label"] == cls]["peak_intensity"].dropna()
            if len(sub_c):
                ax.scatter(
                    xi + jitter(len(sub_c)), sub_c,
                    color=color, alpha=0.35, s=18, linewidths=0, zorder=2,
                )
            # Errors
            sub_e = err_df[err_df["fa_label"] == cls]["peak_intensity"].dropna()
            if len(sub_e):
                ax.scatter(
                    xi + jitter(len(sub_e)), sub_e,
                    color=color, edgecolors="black", linewidths=1.0,
                    s=55, zorder=3, marker="^",
                )

        ax.set_xticks(range(len(classes_present)))
        ax.set_xticklabels(
            [FA5_SHORT[c] for c in classes_present], fontsize=9
        )
        ax.set_xlabel("FA sub-label", fontsize=9)
        ax.set_ylabel(f"{INTENSITY_PERCENTILE}th pct pixel intensity", fontsize=9)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)

    # Legend
    correct_patch = mpatches.Patch(color="gray", alpha=0.5, label="Correct (circle)")
    error_patch   = plt.Line2D([0], [0], marker="^", color="gray",
                                markeredgecolor="black", markersize=8,
                                linewidth=0, label="Error (triangle)")
    fig.legend(handles=[correct_patch, error_patch], loc="lower center",
               ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        f"Peak intensity ({INTENSITY_PERCENTILE}th pct) by FA sub-label  [{cfg_label}]\n"
        "Triangles = misclassified  |  Circles = correct",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=int, choices=[0, 1, 2], default=None,
                    help="Config ID (0=s1v3, 1=s2v2, 2=s3v1). Default: run all three.")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    cfg_ids = [args.cfg] if args.cfg is not None else [0, 1, 2]

    for cfg_id in cfg_ids:
        cfg_label = CONFIGS[cfg_id]["label"]
        print(f"\n[cfg {cfg_id} = {cfg_label}]")

        pred_df, bacc = build_predictions(cfg_id)
        pred_df = attach_fa4(pred_df)
        pred_df = attach_intensity(pred_df)

        pred_csv = OUT_DIR / f"noad_ad_error_predictions_cfg{cfg_id}.csv"
        pred_df.to_csv(pred_csv, index=False)
        print(f"Saved: {pred_csv}")

        n_total = len(pred_df)
        n_err   = (~pred_df["correct"]).sum()
        print(f"Test set: {n_total}  errors: {n_err} ({n_err/n_total*100:.1f}%)")

        plot_error_grid(pred_df, OUT_DIR / f"noad_ad_errors_grid_cfg{cfg_id}.png",
                        cfg_label=f"{cfg_label}  BAcc={bacc:.3f}")
        plot_intensity(pred_df,  OUT_DIR / f"noad_ad_errors_intensity_cfg{cfg_id}.png",
                       cfg_label=f"{cfg_label}  BAcc={bacc:.3f}")


if __name__ == "__main__":
    main()
