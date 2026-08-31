#!/usr/bin/env python3
"""
compute_le_b2_umap_grid.py

UMAP grid for the B2 label-efficiency SupCon-AE benchmark.

Grid layout (per dataset):
  Rows    = 5 CV folds
  Columns = selected budgets  (repeat=0 for all panels)

Color scheme:
  dark blue  / dark red   — train-fold labeled adhesion / no-adhesion
  light blue / light red  — test-fold labeled adhesion / no-adhesion
  gray                    — unlabeled background (subsampled to BG_N)

Usage
-----
  python scripts/compute_le_b2_umap_grid.py --dataset ds1
  python scripts/compute_le_b2_umap_grid.py --dataset ds2
  python scripts/compute_le_b2_umap_grid.py --dataset ds3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import umap

REPO    = Path(__file__).resolve().parents[1]
DATA    = Path("/net/projects/CLS/lding/data/fa_data_analysis")
RUN_ROOT = DATA / "ae_results/contrastive_run/le_b2_supcon"
ANN_ROOT = DATA / "labelling/le_b2_supcon"

Z_COLS = [f"z_{i}" for i in range(32)]

# Budgets to display as columns per dataset
DISPLAY_BUDGETS = {
    "ds1": ["10", "50", "150", "300", "750", "all"],
    "ds2": ["10", "25", "75", "150", "all"],
    "ds3": ["10", "25", "75", "150", "all"],
}

DS_TITLE = {
    "ds1": "DS1 (vinc, control+ycomp)",
    "ds2": "DS2 (pfak, control+ycomp)",
    "ds3": "DS3 (ppax, control+ycomp)",
}

N_FOLDS = 5
REPEAT  = 0
BG_N    = 2000
UMAP_NN    = 15
UMAP_MDIST = 0.1

C_ADHE_TRAIN = "#2166AC"   # dark blue
C_NOAD_TRAIN = "#D6604D"   # dark red
C_ADHE_TEST  = "#92C5DE"   # light blue
C_NOAD_TEST  = "#F4A582"   # light red / salmon
C_BG         = "#DDDDDD"   # gray


def _hyph_to_under(uid: str) -> str:
    """Convert hyphenated annotation key → underscore patch filename."""
    return uid.replace("-f", "_f", 1)


def make_panel(ax, ds: str, fold: int, budget: str,
               fold_splits: pd.DataFrame) -> bool:
    """Draw one UMAP panel. Returns True if data was found."""
    nb_str  = budget if budget == "all" else budget
    name    = f"le_b2_{ds}_fv{fold}_nb{nb_str}_r{REPEAT}"
    run_dir = RUN_ROOT / name
    ann_csv = ANN_ROOT / f"{name}.csv"

    if not (run_dir / "latents.csv").exists():
        ax.text(0.5, 0.5, "pending", ha="center", va="center",
                transform=ax.transAxes, fontsize=7, color="#AAAAAA")
        ax.axis("off")
        return False

    lat = pd.read_csv(run_dir / "latents.csv")

    # Load train annotation (unique_ID hyphenated → convert to underscore for merge)
    train_ann = pd.read_csv(ann_csv)
    train_ann["filename"] = train_ann["unique_ID"].apply(_hyph_to_under)

    # Test-fold labels from fold_splits
    test_fold = fold_splits[fold_splits["fold"] == fold].copy()
    test_fold["filename"] = test_fold["unique_ID"].apply(_hyph_to_under)

    # Merge train labels
    lat = lat.merge(
        train_ann[["filename", "label"]].rename(columns={"label": "train_label"}),
        on="filename", how="left"
    )
    # Merge test labels
    lat = lat.merge(
        test_fold[["filename", "label"]].rename(columns={"label": "test_label"}),
        on="filename", how="left"
    )

    is_train = lat["train_label"].notna()
    is_test  = lat["test_label"].notna() & ~is_train
    is_bg    = ~is_train & ~is_test

    # Subsample background
    bg_idx = lat[is_bg].index.tolist()
    rng    = np.random.default_rng(42)
    if len(bg_idx) > BG_N:
        bg_idx = rng.choice(bg_idx, BG_N, replace=False).tolist()

    keep = sorted(set(lat[is_train].index.tolist() +
                      lat[is_test].index.tolist() +
                      bg_idx))
    sub = lat.loc[keep].copy().reset_index(drop=True)

    # Recompute masks after subsetting
    is_tr = sub["train_label"].notna()
    is_te = sub["test_label"].notna() & ~is_tr
    is_bg2 = ~is_tr & ~is_te

    # UMAP
    X = sub[Z_COLS].values
    reducer = umap.UMAP(n_neighbors=UMAP_NN, min_dist=UMAP_MDIST,
                        metric="euclidean", random_state=42, verbose=False)
    emb = reducer.fit_transform(X)
    sub[["u1", "u2"]] = emb

    # Plot: background → test → train (train on top)
    ax.scatter(sub.loc[is_bg2, "u1"], sub.loc[is_bg2, "u2"],
               s=1, c=C_BG, alpha=0.5, linewidths=0, rasterized=True)

    for lbl, col in [("adhesion", C_ADHE_TEST), ("No adhesion", C_NOAD_TEST)]:
        m = is_te & (sub["test_label"] == lbl)
        if m.any():
            ax.scatter(sub.loc[m, "u1"], sub.loc[m, "u2"],
                       s=8, c=col, alpha=0.7, linewidths=0)

    for lbl, col in [("adhesion", C_ADHE_TRAIN), ("No adhesion", C_NOAD_TRAIN)]:
        m = is_tr & (sub["train_label"] == lbl)
        if m.any():
            ax.scatter(sub.loc[m, "u1"], sub.loc[m, "u2"],
                       s=18, c=col, alpha=1.0, linewidths=0.3,
                       edgecolors="white")

    n_tr = is_tr.sum()
    ax.set_title(f"n={n_tr}", fontsize=7, pad=2)
    ax.axis("off")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ds1", "ds2", "ds3"], required=True)
    args = ap.parse_args()
    ds = args.dataset

    budgets = DISPLAY_BUDGETS[ds]
    n_cols  = len(budgets)
    n_rows  = N_FOLDS

    fold_splits = pd.read_csv(ANN_ROOT / f"fold_splits_{ds}.csv")

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.5 * n_cols, 2.5 * n_rows),
        facecolor="white",
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for ri in range(n_rows):
        for ci, budget in enumerate(budgets):
            ax = axes[ri, ci]
            make_panel(ax, ds, fold=ri, budget=budget, fold_splits=fold_splits)
            if ri == 0:
                label = f"budget={budget}"
                ax.set_title(label, fontsize=8.5, fontweight="bold", pad=4)
            if ci == 0:
                ax.set_ylabel(f"Fold {ri}", fontsize=8, labelpad=4)
                ax.axis("on")
                ax.set_yticks([])
                ax.set_xticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

    legend_els = [
        mpatches.Patch(color=C_ADHE_TRAIN, label="Train adhesion"),
        mpatches.Patch(color=C_NOAD_TRAIN, label="Train no-adhesion"),
        mpatches.Patch(color=C_ADHE_TEST,  label="Test adhesion"),
        mpatches.Patch(color=C_NOAD_TEST,  label="Test no-adhesion"),
        mpatches.Patch(color=C_BG,         label="Unlabeled background"),
    ]
    fig.legend(handles=legend_els, loc="lower center", ncol=5,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f"UMAP — SupCon-AE latents  {DS_TITLE[ds]}  (repeat={REPEAT})\n"
        "dark=train labels,  light=test labels,  gray=unlabeled",
        fontsize=10, fontweight="bold", y=1.01,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    out = REPO / "results" / f"le_b2_umap_grid_{ds}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
