#!/usr/bin/env python3
"""
plot_umap_intensity.py
======================
UMAP of the s2v2 latent space (le_c1_npi100_r0).

Each point:
  fill colour  = peak intensity (99th pct pixel value) — blue colormap
  edge colour  = FA 5-class label (labeled patches only)
    No adhesion        → purple  #8B3FC8
    Nascent Adhesion   → blue    #1f77b4
    focal complex      → orange  #ff7f0e
    focal adhesion     → green   #2ca02c
    fibrillar adhesion → red     #d62728

Unlabeled patches: thin gray edge, small size.
Labeled patches: thick colored edge, larger size, plotted on top.

Output: results/umap_intensity_s2v2.png
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from umap import UMAP

# ---------------------------------------------------------------------------
DATA   = Path("/net/projects/CLS/lding/data/fa_data_analysis")
LAB    = DATA / "labelling"
EVAL   = DATA / "ae_results/contrastive_run/fa4_xds_eval"
LE_DIR = DATA / "ae_results/contrastive_run/le_clean"

RUN_DIR  = LE_DIR / "le_c1_npi100_r0"
FULL_4CLS = LAB / "vinc_control_label_Annabel_20260715_1554.csv"
INT_CSV   = EVAL / "patch_intensities_all.csv"

OUT = _REPO / "results" / "umap_intensity_s2v2.png"

Z_COLS = [f"z_{i}" for i in range(12)]

# Edge colour per FA class (as specified)
CLASS_EDGE = {
    "No adhesion":        "#8B3FC8",   # purple
    "Nascent Adhesion":   "#1f77b4",   # blue
    "focal complex":      "#ff7f0e",   # orange
    "focal adhesion":     "#2ca02c",   # green
    "fibrillar adhesion": "#d62728",   # red
}
CLASS_LABEL = {
    "No adhesion":        "No adhesion",
    "Nascent Adhesion":   "Nascent Adhesion (NA)",
    "focal complex":      "Focal Complex (FC)",
    "focal adhesion":     "Focal Adhesion (FA)",
    "fibrillar adhesion": "Fibrillar Adhesion (Fib)",
}
CLASS_ORDER = ["No adhesion", "Nascent Adhesion", "focal complex",
               "focal adhesion", "fibrillar adhesion"]


def main():
    print("Loading latents ...")
    lat = pd.read_csv(RUN_DIR / "latents.csv")

    print("Loading intensities ...")
    int_df = pd.read_csv(INT_CSV)
    int_df = int_df[int_df["dataset"] == "vinc_ctrl"][["filename", "peak_intensity"]]
    lat = lat.merge(int_df, on="filename", how="left")

    print("Loading 4-class labels ...")
    lab4 = pd.read_csv(FULL_4CLS)[["filename", "label"]].rename(
        columns={"label": "fa_label"}
    )
    # Add No adhesion from labeled set
    full2_path = LAB / "vinc_control_label_Annabel_20260715_1554_2cls.csv"
    lab2 = pd.read_csv(full2_path)[["filename", "label"]].rename(
        columns={"label": "label_2cls"}
    )
    # Merge: get fa_label for all labeled patches
    all_labels = lab2.merge(lab4, on="filename", how="left")
    all_labels["fa_label"] = all_labels.apply(
        lambda r: "No adhesion" if r["label_2cls"] == "No adhesion"
        else r["fa_label"], axis=1
    )
    lat = lat.merge(all_labels[["filename", "fa_label"]], on="filename", how="left")

    print(f"Total patches: {len(lat)}")
    print(f"Labeled: {lat['fa_label'].notna().sum()}")
    print("Label breakdown:", lat["fa_label"].value_counts().to_dict())

    # UMAP
    print("Running UMAP ...")
    Z = lat[Z_COLS].values
    reducer = UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                   metric="euclidean", random_state=42)
    emb = reducer.fit_transform(Z)
    lat["umap_x"] = emb[:, 0]
    lat["umap_y"] = emb[:, 1]

    # Normalise intensity for colormap
    p_lo = np.percentile(lat["peak_intensity"].dropna(), 2)
    p_hi = np.percentile(lat["peak_intensity"].dropna(), 98)
    norm = mcolors.Normalize(vmin=p_lo, vmax=p_hi, clip=True)
    cmap = cm.Blues

    fig, ax = plt.subplots(figsize=(10, 8), facecolor="white")

    # 1) Unlabeled patches — small, gray edge, blue fill
    unlabeled = lat[lat["fa_label"].isna()]
    facecolors_u = cmap(norm(unlabeled["peak_intensity"].values))
    ax.scatter(
        unlabeled["umap_x"], unlabeled["umap_y"],
        c=facecolors_u, edgecolors="#aaaaaa", linewidths=0.3,
        s=8, alpha=0.6, zorder=1,
    )

    # 2) Labeled patches — larger, colored edge by class, on top
    labeled = lat[lat["fa_label"].notna()]
    for cls in CLASS_ORDER:
        sub = labeled[labeled["fa_label"] == cls]
        if len(sub) == 0:
            continue
        facecolors_l = cmap(norm(sub["peak_intensity"].values))
        ax.scatter(
            sub["umap_x"], sub["umap_y"],
            c=facecolors_l,
            edgecolors=CLASS_EDGE[cls],
            linewidths=1.4,
            s=55, alpha=0.92, zorder=3,
            label=f"{CLASS_LABEL[cls]} (n={len(sub)})",
        )

    # Colourbar for intensity
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("99th pct pixel intensity (fill)", fontsize=10)

    # Legend for class edges
    handles = [
        mpatches.Patch(facecolor="white", edgecolor=CLASS_EDGE[c],
                       linewidth=2, label=CLASS_LABEL[c])
        for c in CLASS_ORDER if (labeled["fa_label"] == c).any()
    ]
    handles.append(
        mpatches.Patch(facecolor="white", edgecolor="#aaaaaa",
                       linewidth=1, label="Unlabeled patches")
    )
    ax.legend(handles=handles, fontsize=9, loc="upper left",
              framealpha=0.85, edgecolor="#cccccc")

    ax.set_xlabel("UMAP 1", fontsize=11)
    ax.set_ylabel("UMAP 2", fontsize=11)
    ax.set_title(
        "UMAP — s2v2 latent space (le_c1_npi100_r0)\n"
        "Fill = 99th-pct intensity (blue scale) · Edge = FA class label",
        fontsize=11, fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("white")

    fig.tight_layout()
    fig.savefig(str(OUT), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
