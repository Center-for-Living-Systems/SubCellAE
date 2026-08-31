#!/usr/bin/env python3
"""
plot_cp_feature_histograms_ds1.py

Histogram of all 61 CellProfiler features for DS1, grouped by type.
Groups: raw intensity (11), cioprt intensity (11), Haralick d1/d2/d4 (13×3=39)

Output: results/cp_feature_histograms_ds1.png
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

DATA     = Path("/net/projects/CLS/lding/data/fa_data_analysis")
FEAT_DIR = DATA / "ae_results/features/cellprofiler"
REPO     = Path(__file__).resolve().parents[1]
OUT      = REPO / "results" / "cp_feature_histograms_ds1.png"

N_BINS = 50

GROUP_COLORS = {
    "intensity_raw":   "#E8F4FD",
    "intensity_cioprt":"#FEF9E7",
    "haralick_d1":     "#F9EBEA",
    "haralick_d2":     "#EAFAF1",
    "haralick_d4":     "#F5EEF8",
}

def group_of(col: str) -> str:
    if col.endswith("_cioprt"):
        return "intensity_cioprt"
    if col.startswith("intensity_"):
        return "intensity_raw"
    if col.endswith("_d1"):
        return "haralick_d1"
    if col.endswith("_d2"):
        return "haralick_d2"
    return "haralick_d4"


def main():
    df = pd.read_csv(FEAT_DIR / "ds1.csv")
    feat_cols = [c for c in df.columns if c != "filename"]

    # Reorder: raw intensity, cioprt intensity, haralick d1, d2, d4
    order = (
        [c for c in feat_cols if c.startswith("intensity_") and not c.endswith("_cioprt")] +
        [c for c in feat_cols if c.endswith("_cioprt")] +
        [c for c in feat_cols if c.endswith("_d1")] +
        [c for c in feat_cols if c.endswith("_d2")] +
        [c for c in feat_cols if c.endswith("_d4")]
    )

    n_feat = len(order)
    n_cols = 11
    n_rows = (n_feat + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.0, n_rows * 1.8),
                             facecolor="white")
    axes = axes.ravel()

    for fi, col in enumerate(order):
        ax   = axes[fi]
        grp  = group_of(col)
        bg   = GROUP_COLORS[grp]
        vals = df[col].dropna().values

        lo, hi = np.percentile(vals, 1), np.percentile(vals, 99)
        clipped = np.clip(vals, lo, hi)

        ax.set_facecolor(bg)
        ax.hist(clipped, bins=N_BINS, color="#2166AC", alpha=0.75,
                density=True, linewidth=0)

        # Short label: strip _cioprt, _d1/2/4 suffixes for readability
        short = col.replace("_cioprt", "†").replace("intensity_", "int_")
        ax.set_title(short, fontsize=5.2, pad=2)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=4.5)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)

    for fi in range(n_feat, len(axes)):
        axes[fi].set_visible(False)

    # Group legend
    legend_els = [
        mpatches.Patch(color=GROUP_COLORS["intensity_raw"],    label="Raw CIO intensity (11)"),
        mpatches.Patch(color=GROUP_COLORS["intensity_cioprt"], label="CIO-prt intensity† (11)"),
        mpatches.Patch(color=GROUP_COLORS["haralick_d1"],      label="Haralick d=1 (13)"),
        mpatches.Patch(color=GROUP_COLORS["haralick_d2"],      label="Haralick d=2 (13)"),
        mpatches.Patch(color=GROUP_COLORS["haralick_d4"],      label="Haralick d=4 (13)"),
    ]
    fig.legend(handles=legend_els, loc="lower center", ncol=5,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "DS1 (vinc) — CellProfiler features, 61 total  [1–99th pct clipped]\n"
        "† = cio_mode_prt normalization (background mode subtracted, divided by P97.5–P99.5 cell signal)",
        fontsize=9, fontweight="bold", y=1.01,
    )
    fig.tight_layout(pad=0.5, h_pad=0.8, w_pad=0.4)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
