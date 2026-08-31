#!/usr/bin/env python3
"""
analyze_fa4_intensity.py
========================
Check peak intensity distribution of FA patches.

Computes the 99th-percentile pixel intensity per patch (a proxy for "brightness")
and plots:
  1. Distribution per dataset (all patches, not just labeled)
  2. Distribution per FA class (labeled patches only)
  3. Retention curve: fraction of labeled patches retained vs intensity threshold

Saves:
  intensity_distribution.png — all three panels

Usage
-----
  python scripts/analyze_fa4_intensity.py
  python scripts/analyze_fa4_intensity.py --sample 5000   # subsample to speed up
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

# ---------------------------------------------------------------------------
DATA_ROOT = Path("/net/projects/CLS/lding/data/fa_data_analysis")
EVAL_DIR  = DATA_ROOT / "ae_results" / "contrastive_run" / "fa4_xds_eval"
LABEL_DIR = DATA_ROOT / "labelling"

PATCH_DIRS = {
    "vinc_ctrl":  DATA_ROOT / "ae_results" / "patches" / "cio"    / "vinc" / "control" / "tiff_patches32_mr10",
    "vinc_ycomp": DATA_ROOT / "ae_results" / "patches" / "cio"    / "vinc" / "ycomp"   / "tiff_patches32_mr10",
    "pfak_ctrl":  DATA_ROOT / "ae_results" / "patches" / "cio_rb" / "pfak" / "control" / "tiff_patches32_mr10",
}

LABEL_FILES = {
    "vinc_ctrl":  LABEL_DIR / "vinc_control_label_Annabel_20260715_1554.csv",
    "vinc_ycomp": LABEL_DIR / "vinc_combined_label_Annabel_20260816.csv",
    "pfak_ctrl":  LABEL_DIR / "pfak_combined_label_Annabel_aug2026.csv",
}

FA_LABEL_ORDER_4 = [
    "Nascent Adhesion",
    "focal complex",
    "focal adhesion",
    "fibrillar adhesion",
]
LABEL_SHORT = {
    "Nascent Adhesion":   "NA",
    "focal complex":      "FC",
    "focal adhesion":     "FA",
    "fibrillar adhesion": "Fib",
}
LABEL_COLORS = {
    "No adhesion":        "#9467bd",  # purple
    "Nascent Adhesion":   "#1565C0",  # blue
    "focal complex":      "#E65100",  # orange
    "focal adhesion":     "#2ca02c",  # green
    "fibrillar adhesion": "#C00000",  # red
}
DS_COLORS = {
    "vinc_ctrl":  "#1f77b4",
    "vinc_ycomp": "#ff7f0e",
    "pfak_ctrl":  "#d62728",
}

INTENSITY_PERCENTILE = 99  # "peak" intensity = 99th pct pixel value in patch


def _peak_intensity(img: np.ndarray) -> float:
    return float(np.percentile(img, INTENSITY_PERCENTILE))


def compute_all_intensities(sample: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Compute peak intensity for all patches in all datasets."""
    rng = np.random.default_rng(seed)
    rows = []
    for ds_key, patch_dir in PATCH_DIRS.items():
        tifs = sorted(patch_dir.glob("*.tif"))
        print(f"[{ds_key}] {len(tifs)} patches")
        if sample and len(tifs) > sample:
            idxs = rng.choice(len(tifs), size=sample, replace=False)
            tifs = [tifs[i] for i in sorted(idxs)]

        for p in tifs:
            img = tifffile.imread(str(p)).astype(np.float32)
            rows.append({"dataset": ds_key, "filename": p.name,
                         "peak_intensity": _peak_intensity(img)})

    return pd.DataFrame(rows)


def compute_labeled_intensities() -> pd.DataFrame:
    """Compute peak intensity only for labeled patches, with class column."""
    rows = []
    for ds_key, patch_dir in PATCH_DIRS.items():
        lab_df = pd.read_csv(LABEL_FILES[ds_key])
        lab_df = lab_df[lab_df["label"].isin(FA_LABEL_ORDER_4)][["filename", "label"]]
        print(f"[{ds_key}] {len(lab_df)} labeled patches")
        for _, row in lab_df.iterrows():
            p = patch_dir / row["filename"]
            if not p.exists():
                continue
            img = tifffile.imread(str(p)).astype(np.float32)
            rows.append({"dataset": ds_key, "filename": row["filename"],
                         "label": row["label"],
                         "peak_intensity": _peak_intensity(img)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None,
                        help="Max patches per dataset for the full-set plot (default: all)")
    args = parser.parse_args()

    print("[step 1] Computing intensities for all patches ...")
    df_all = compute_all_intensities(sample=args.sample)

    print("[step 2] Computing intensities for labeled patches ...")
    df_lab = compute_labeled_intensities()

    # Save intensity tables
    df_all.to_csv(EVAL_DIR / "patch_intensities_all.csv", index=False)
    df_lab.to_csv(EVAL_DIR / "patch_intensities_labeled.csv", index=False)
    print(f"[saved] intensity CSVs → {EVAL_DIR}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

    # Panel 1: per-dataset histogram (all patches)
    ax = axes[0]
    for ds_key in PATCH_DIRS:
        sub = df_all[df_all["dataset"] == ds_key]["peak_intensity"].values
        if len(sub) == 0:
            continue
        ax.hist(sub, bins=80, density=True, alpha=0.5, color=DS_COLORS[ds_key],
                label=ds_key.replace("_", " "), edgecolor="none")
    ax.set_xlabel(f"{INTENSITY_PERCENTILE}th pct pixel value", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title("Peak intensity — all patches\n(full dataset)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel 2: per-class violin (labeled patches only) — always show all 4 classes
    ax = axes[1]
    data_by_class = [
        df_lab[df_lab["label"] == cls]["peak_intensity"].values
        for cls in FA_LABEL_ORDER_4
    ]
    nonempty_idx = [(i, cls, d) for i, (cls, d) in enumerate(zip(FA_LABEL_ORDER_4, data_by_class)) if len(d) > 0]
    if nonempty_idx:
        vp = ax.violinplot(
            [d for _, _, d in nonempty_idx],
            positions=[i for i, _, _ in nonempty_idx],
            showmedians=True, showextrema=True,
        )
        for body, (_, cls, _) in zip(vp["bodies"], nonempty_idx):
            body.set_facecolor(LABEL_COLORS[cls])
            body.set_alpha(0.6)
        vp["cmedians"].set_color("black")
    # Mark empty classes
    for i, (cls, d) in enumerate(zip(FA_LABEL_ORDER_4, data_by_class)):
        if len(d) == 0:
            ax.text(i, 0, "n=0", ha="center", va="bottom", fontsize=8,
                    color=LABEL_COLORS[cls], fontweight="bold")
    ax.set_xticks(range(len(FA_LABEL_ORDER_4)))
    ax.set_xticklabels([LABEL_SHORT[c] for c in FA_LABEL_ORDER_4], fontsize=9)
    ax.set_xlabel("FA class", fontsize=9)
    ax.set_ylabel(f"{INTENSITY_PERCENTILE}th pct pixel value", fontsize=9)
    ax.set_title("Peak intensity by class\n(labeled patches only)", fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    # Panel 3: retention curve — % labeled patches retained vs threshold
    ax = axes[2]
    all_intensities = df_lab["peak_intensity"].values
    thresholds = np.linspace(all_intensities.min(), all_intensities.max(), 200)
    for cls in FA_LABEL_ORDER_4:
        sub = df_lab[df_lab["label"] == cls]["peak_intensity"].values
        if len(sub) == 0:
            ax.plot([], [], color=LABEL_COLORS[cls], linewidth=1.8,
                    linestyle=":", label=f"{LABEL_SHORT[cls]} (n=0)")
            continue
        retention = [np.mean(sub >= t) * 100 for t in thresholds]
        ax.plot(thresholds, retention, color=LABEL_COLORS[cls],
                linewidth=1.8, label=f"{LABEL_SHORT[cls]} (n={len(sub)})")

    # Also show overall retention
    overall = [np.mean(all_intensities >= t) * 100 for t in thresholds]
    ax.plot(thresholds, overall, "k--", linewidth=1.2, label="All classes", alpha=0.7)

    # Mark 50th/75th/90th pct of all intensities
    for pct in [50, 75, 90]:
        thresh = np.percentile(all_intensities, pct)
        ax.axvline(thresh, color="gray", linewidth=0.8, linestyle=":",
                   label=f"p{pct}={thresh:.2f}")

    ax.set_xlabel(f"Intensity threshold (min {INTENSITY_PERCENTILE}th pct peak)", fontsize=9)
    ax.set_ylabel("% patches retained", fontsize=9)
    ax.set_title("Retention by class vs threshold\n(labeled patches)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 105)

    fig.suptitle(
        "FA patch peak-intensity analysis\n"
        f"({INTENSITY_PERCENTILE}th percentile pixel value per 32×32 patch)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    out = EVAL_DIR / "intensity_distribution.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
