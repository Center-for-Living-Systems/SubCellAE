#!/usr/bin/env python3
"""
plot_cp_feature_histograms.py

One PNG: 50 CellProfiler features, DS1/DS2/DS3 overlaid per subplot.
Features grouped by type with background shading.

Output: results/cp_feature_histograms.png
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
OUT      = REPO / "results" / "cp_feature_histograms.png"

DS_COLORS = {"ds1": "#2166AC", "ds2": "#D6604D", "ds3": "#4DAC26"}
DS_LABELS = {"ds1": "DS1 (vinc)", "ds2": "DS2 (pfak)", "ds3": "DS3 (ppax)"}
N_BINS    = 50
N_COLS    = 10

GROUP_BG = {
    "intensity":   "#E8F4FD",
    "haralick_d1": "#F9EBEA",
    "haralick_d2": "#EAFAF1",
    "haralick_d4": "#F5EEF8",
}

def group_of(col):
    if col.startswith("intensity_"): return "intensity"
    if col.endswith("_d1"):          return "haralick_d1"
    if col.endswith("_d2"):          return "haralick_d2"
    return "haralick_d4"


def main():
    dfs = {ds: pd.read_csv(FEAT_DIR / f"{ds}.csv") for ds in DS_COLORS}
    feat_cols = [c for c in next(iter(dfs.values())).columns if c != "filename"]

    order = (
        [c for c in feat_cols if c.startswith("intensity_")] +
        [c for c in feat_cols if c.endswith("_d1")] +
        [c for c in feat_cols if c.endswith("_d2")] +
        [c for c in feat_cols if c.endswith("_d4")]
    )

    n_rows = (len(order) + N_COLS - 1) // N_COLS
    fig, axes = plt.subplots(n_rows, N_COLS,
                             figsize=(N_COLS * 2.2, n_rows * 1.8),
                             facecolor="white")
    axes = axes.ravel()

    for fi, col in enumerate(order):
        ax = axes[fi]
        ax.set_facecolor(GROUP_BG[group_of(col)])

        for ds, df in dfs.items():
            vals    = df[col].dropna().values
            lo, hi  = np.percentile(vals, 1), np.percentile(vals, 99)
            clipped = np.clip(vals, lo, hi)
            ax.hist(clipped, bins=N_BINS, color=DS_COLORS[ds],
                    alpha=0.5, density=True, linewidth=0)

        short = col.replace("intensity_", "int_")
        ax.set_title(short, fontsize=5.5, pad=2)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=4.8)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)

    for fi in range(len(order), len(axes)):
        axes[fi].set_visible(False)

    ds_handles  = [mpatches.Patch(color=DS_COLORS[ds], alpha=0.6, label=DS_LABELS[ds])
                   for ds in DS_COLORS]
    grp_handles = [mpatches.Patch(color=v, label=k.replace("_", " ").replace("haralick ", "Haralick ").replace("intensity", "Intensity (11)").replace("Haralick d1", "Haralick d=1 (13)").replace("Haralick d2", "Haralick d=2 (13)").replace("Haralick d4", "Haralick d=4 (13)"))
                   for k, v in GROUP_BG.items()]
    fig.legend(handles=ds_handles + grp_handles, loc="lower center", ncol=7,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("CellProfiler features (50 total) — DS1 / DS2 / DS3 overlaid  [1–99th pct clipped]",
                 fontsize=10, fontweight="bold", y=1.005)
    fig.tight_layout(pad=0.5, h_pad=0.8, w_pad=0.4)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
