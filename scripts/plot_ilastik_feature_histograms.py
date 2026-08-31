#!/usr/bin/env python3
"""
plot_ilastik_feature_histograms.py

One PNG: 80 ilastik features, DS1/DS2/DS3 overlaid per subplot.
Grouped by filter type with background shading.

Output: results/ilastik_feature_histograms.png
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
FEAT_DIR = DATA / "ae_results/features/ilastik"
REPO     = Path(__file__).resolve().parents[1]
OUT      = REPO / "results" / "ilastik_feature_histograms.png"

DS_COLORS = {"ds1": "#2166AC", "ds2": "#D6604D", "ds3": "#4DAC26"}
DS_LABELS = {"ds1": "DS1 (vinc)", "ds2": "DS2 (pfak)", "ds3": "DS3 (ppax)"}
N_BINS    = 50
N_COLS    = 10

FILTER_ORDER = [
    "gaussian", "log", "gradient_mag", "dog",
    "structure_tensor_large", "structure_tensor_small",
    "hessian_large", "hessian_small",
]
GROUP_BG = {
    "gaussian":               "#E8F4FD",
    "log":                    "#FEF9E7",
    "gradient_mag":           "#F9EBEA",
    "dog":                    "#EAFAF1",
    "structure_tensor_large": "#F5EEF8",
    "structure_tensor_small": "#FDF2E9",
    "hessian_large":          "#E9F7EF",
    "hessian_small":          "#FDEDEC",
}
GROUP_LABEL = {
    "gaussian":               "Gaussian",
    "log":                    "LoG",
    "gradient_mag":           "Gradient mag",
    "dog":                    "DoG",
    "structure_tensor_large": "ST large λ",
    "structure_tensor_small": "ST small λ",
    "hessian_large":          "Hessian large λ",
    "hessian_small":          "Hessian small λ",
}

def group_of(col):
    for g in FILTER_ORDER:
        if col.startswith(g):
            return g
    return "gaussian"


def main():
    dfs = {ds: pd.read_csv(FEAT_DIR / f"{ds}.csv") for ds in DS_COLORS}
    feat_cols = [c for c in next(iter(dfs.values())).columns if c != "filename"]

    # Order by filter type then scale
    order = []
    for g in FILTER_ORDER:
        order += [c for c in feat_cols if c.startswith(g)]

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

        # Short label: keep stat symbol + scale
        short = col
        for g in FILTER_ORDER:
            short = short.replace(g + "_", "")
        short = short.replace("_mean_", " μ s").replace("_std_", " σ s").replace("s0p", "").replace("p", ".")
        ax.set_title(short, fontsize=5.2, pad=2)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=4.5)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)

    for fi in range(len(order), len(axes)):
        axes[fi].set_visible(False)

    ds_handles  = [mpatches.Patch(color=DS_COLORS[ds], alpha=0.6, label=DS_LABELS[ds])
                   for ds in DS_COLORS]
    grp_handles = [mpatches.Patch(color=GROUP_BG[g], label=GROUP_LABEL[g])
                   for g in FILTER_ORDER]
    fig.legend(handles=ds_handles + grp_handles, loc="lower center", ncol=6,
               fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Ilastik-style features (80 total) — DS1 / DS2 / DS3 overlaid  [1–99th pct clipped]",
                 fontsize=10, fontweight="bold", y=1.005)
    fig.tight_layout(pad=0.5, h_pad=0.8, w_pad=0.4)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
