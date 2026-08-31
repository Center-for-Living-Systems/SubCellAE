#!/usr/bin/env python3
"""
compute_le_umap_grid.py

For each clean label-efficiency condition (3 cfg × 6 npi, repeat=0),
compute a UMAP of the model's latent space and save a 3×6 grid figure.

  results/le_clean_umap_grid.png

Coloring:
  gray   — unlabeled background patches (subsampled to 2 000)
  blue   — train-frame patches labeled "adhesion"
  red    — train-frame patches labeled "No adhesion"
  lighter blue/red — test-frame patches (ground truth from full Annabel set)
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import umap

REPO   = Path(__file__).resolve().parents[1]
DATA   = Path("/net/projects/CLS/lding/data/fa_data_analysis")
LE_DIR = DATA / "ae_results/contrastive_run/le_clean"
LAB_DIR = DATA / "labelling"
FULL_ANN = LAB_DIR / "vinc_control_label_Annabel_20260715_1554_2cls.csv"

Z_COLS = [f"z_{i}" for i in range(12)]
NPI_ORDER = ["10", "25", "50", "75", "100", "all"]
CFG_FRAMES = {
    0: {"train": [0],       "test": [1, 2, 3]},
    1: {"train": [0, 1],    "test": [2, 3]},
    2: {"train": [0, 1, 2], "test": [3]},
}

C_ADHE_TRAIN  = "#2166AC"   # dark blue  — train adhesion
C_NOAD_TRAIN  = "#D6604D"   # dark red   — train no-adhesion
C_ADHE_TEST   = "#92C5DE"   # light blue — test adhesion
C_NOAD_TEST   = "#F4A582"   # light red  — test no-adhesion
C_BG          = "#DDDDDD"   # gray       — unlabeled background

BG_N       = 2000   # background subsample per panel
UMAP_NN    = 15
UMAP_MDIST = 0.1
RNG        = np.random.default_rng(42)


def extract_frame(filename: str) -> int:
    m = re.search(r"_f(\d+)", filename)
    return int(m.group(1)) if m else -1


def make_panel(ax, cfg: int, npi: str, full_ann: pd.DataFrame) -> str:
    run_name = f"le_c{cfg}_npi{npi}_r0"
    run_dir  = LE_DIR / run_name
    ann_csv  = LAB_DIR / "le_clean" / f"{run_name}.csv"

    if not run_dir.exists() or not (run_dir / "latents.csv").exists():
        ax.text(0.5, 0.5, "missing", ha="center", va="center",
                transform=ax.transAxes, fontsize=7, color="#AAAAAA")
        ax.axis("off")
        return run_name

    lat = pd.read_csv(run_dir / "latents.csv")
    lat["frame"] = lat["filename"].apply(extract_frame)
    ann_train    = pd.read_csv(ann_csv)
    frames       = CFG_FRAMES[cfg]

    # Tag each patch: train-labeled, test-labeled, or background
    lat = lat.merge(ann_train[["filename", "label"]].rename(
                        columns={"label": "train_label"}),
                    on="filename", how="left")
    lat = lat.merge(full_ann[["filename", "label"]].rename(
                        columns={"label": "full_label"}),
                    on="filename", how="left")

    train_mask = lat["frame"].isin(frames["train"]) & lat["train_label"].notna()
    test_mask  = lat["frame"].isin(frames["test"])  & lat["full_label"].notna()
    bg_mask    = ~train_mask & ~test_mask

    # Subsample background
    bg_idx     = lat[bg_mask].index.tolist()
    if len(bg_idx) > BG_N:
        bg_idx = RNG.choice(bg_idx, BG_N, replace=False).tolist()

    keep_idx = sorted(set(lat[train_mask].index.tolist() +
                          lat[test_mask].index.tolist() +
                          bg_idx))
    sub = lat.loc[keep_idx].copy()

    # UMAP
    X = sub[Z_COLS].values
    reducer = umap.UMAP(n_neighbors=UMAP_NN, min_dist=UMAP_MDIST,
                        metric="euclidean", random_state=42, verbose=False)
    emb = reducer.fit_transform(X)

    sub = sub.reset_index(drop=True)
    emb_df = pd.DataFrame(emb, columns=["u1", "u2"])
    sub    = pd.concat([sub.reset_index(drop=True), emb_df], axis=1)

    is_train = sub["frame"].isin(frames["train"]) & sub["train_label"].notna()
    is_test  = sub["frame"].isin(frames["test"])  & sub["full_label"].notna()
    is_bg    = ~is_train & ~is_test

    # Plot background first
    ax.scatter(sub.loc[is_bg, "u1"], sub.loc[is_bg, "u2"],
               s=1, c=C_BG, alpha=0.5, linewidths=0, rasterized=True)

    # Test-frame labeled (lighter)
    for lbl, col in [("adhesion", C_ADHE_TEST), ("No adhesion", C_NOAD_TEST)]:
        m = is_test & (sub["full_label"] == lbl)
        if m.any():
            ax.scatter(sub.loc[m, "u1"], sub.loc[m, "u2"],
                       s=10, c=col, alpha=0.7, linewidths=0, marker="o")

    # Train-frame labeled (darker, larger)
    for lbl, col in [("adhesion", C_ADHE_TRAIN), ("No adhesion", C_NOAD_TRAIN)]:
        m = is_train & (sub["train_label"] == lbl)
        if m.any():
            ax.scatter(sub.loc[m, "u1"], sub.loc[m, "u2"],
                       s=20, c=col, alpha=1.0, linewidths=0.3,
                       edgecolors="white", marker="o")

    n_train = is_train.sum()
    ax.set_title(f"n_labels={n_train}", fontsize=7, pad=2)
    ax.axis("off")
    return run_name


def main():
    full_ann = pd.read_csv(FULL_ANN)

    fig, axes = plt.subplots(
        3, 6,
        figsize=(16, 8),
        facecolor="white",
    )

    cfg_labels = {
        0: "cfg0  train=[0]  test=[1,2,3]",
        1: "cfg1  train=[0,1]  test=[2,3]",
        2: "cfg2  train=[0,1,2]  test=[3]",
    }

    for ri, cfg in enumerate([0, 1, 2]):
        for ci, npi in enumerate(NPI_ORDER):
            ax = axes[ri, ci]
            run = make_panel(ax, cfg, npi, full_ann)
            print(f"  {run}")
            if ci == 0:
                ax.set_ylabel(cfg_labels[cfg], fontsize=8, labelpad=4)
        axes[ri, 0].axis("on")
        axes[ri, 0].set_yticks([])
        axes[ri, 0].set_xticks([])
        for spine in axes[ri, 0].spines.values():
            spine.set_visible(False)

    for ci, npi in enumerate(NPI_ORDER):
        axes[0, ci].set_title(f"n_per_img={npi}", fontsize=8.5, fontweight="bold", pad=4)

    # Legend
    legend_els = [
        mpatches.Patch(color=C_ADHE_TRAIN, label="Train adhesion"),
        mpatches.Patch(color=C_NOAD_TRAIN, label="Train no-adhesion"),
        mpatches.Patch(color=C_ADHE_TEST,  label="Test adhesion (Annabel)"),
        mpatches.Patch(color=C_NOAD_TEST,  label="Test no-adhesion (Annabel)"),
        mpatches.Patch(color=C_BG,         label="Unlabeled background"),
    ]
    fig.legend(handles=legend_els, loc="lower center", ncol=5,
               fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        "UMAP of clean SupCon AE latents  —  repeat=0,  each panel is a separately trained model\n"
        "dark=train labels,  light=test labels (Annabel ground truth),  gray=unlabeled",
        fontsize=10, fontweight="bold", y=1.01,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    out = REPO / "results" / "le_clean_umap_grid.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
