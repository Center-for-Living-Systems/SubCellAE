#!/usr/bin/env python3
"""
compute_cp_il_umap.py

UMAP of CellProfiler and ilastik features for DS1 (vinc ctrl+ycomp),
using the same color scheme as the SupCon-AE LE benchmark UMAPs.

Color scheme (fold=0 reference):
  dark blue  / dark red   — train-fold B2-labeled adhesion / no-adhesion
  light blue / light red  — test-fold  B2-labeled adhesion / no-adhesion
  gray                    — all other patches (subsampled for clarity)

Saves:
  results/umap_cp_ds1.png
  results/umap_ilastik_ds1.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import umap
from sklearn.preprocessing import StandardScaler

DATA    = Path("/net/projects/CLS/lding/data/fa_data_analysis")
REPO    = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "results"

ANN_FILE    = DATA / "labelling/vinc_combined_label_Annabel_20260816.csv"
FOLD_SPLITS = DATA / "labelling/le_b2_supcon/fold_splits_ds1.csv"

FEAT_FILES = {
    "cp":      DATA / "ae_results/features/cellprofiler/ds1.csv",
    "ilastik": DATA / "ae_results/features/ilastik/ds1.csv",
}

FOLD   = 0
BG_N   = 3000
UMAP_NN    = 15
UMAP_MDIST = 0.1

C_ADHE_TRAIN = "#2166AC"
C_NOAD_TRAIN = "#D6604D"
C_ADHE_TEST  = "#92C5DE"
C_NOAD_TEST  = "#F4A582"
C_BG         = "#CCCCCC"

LABEL_NAMES = {
    "adhesion":    ("Adhesion (train)",    "Adhesion (test)"),
    "No adhesion": ("No adhesion (train)", "No adhesion (test)"),
}


def _h2u(uid: str) -> str:
    return uid.replace("-f", "_f", 1)


def _binarize(label: str) -> str:
    return "No adhesion" if label == "No adhesion" else "adhesion"


def make_umap(method: str, feat_path: Path):
    print(f"\n=== {method} ===")

    # ---- load features ----
    feat = pd.read_csv(feat_path)
    feat_cols = [c for c in feat.columns if c != "filename"]
    print(f"  {len(feat)} patches, {len(feat_cols)} features")

    # ---- load B2 annotations ----
    ann = pd.read_csv(ANN_FILE)
    ann = ann[ann["filename"].str.startswith(("control_", "ycomp_"))].copy()
    ann["label"]    = ann["label"].apply(_binarize)
    ann["filename"] = ann["filename"].apply(_h2u)

    # ---- load fold splits ----
    splits = pd.read_csv(FOLD_SPLITS)
    splits["filename"] = splits["unique_ID"].apply(_h2u)
    test_fns  = set(splits[splits["fold"] == FOLD]["filename"])
    train_fns = set(ann["filename"]) - test_fns

    # ---- merge labels onto features ----
    ann_map = dict(zip(ann["filename"], ann["label"]))
    feat["_label"]    = feat["filename"].map(ann_map)
    feat["_is_train"] = feat["filename"].isin(train_fns) & feat["_label"].notna()
    feat["_is_test"]  = feat["filename"].isin(test_fns)  & feat["_label"].notna()
    feat["_is_bg"]    = ~feat["_is_train"] & ~feat["_is_test"]

    # subsample background
    bg_idx = feat[feat["_is_bg"]].index.tolist()
    rng    = np.random.default_rng(42)
    if len(bg_idx) > BG_N:
        bg_idx = list(rng.choice(bg_idx, BG_N, replace=False))

    keep_idx = sorted(
        feat[feat["_is_train"]].index.tolist() +
        feat[feat["_is_test"]].index.tolist() +
        bg_idx
    )
    sub = feat.loc[keep_idx].copy().reset_index(drop=True)

    # ---- UMAP ----
    X = sub[feat_cols].values.astype(float)
    X = np.nan_to_num(X)
    X = StandardScaler().fit_transform(X)

    print(f"  Running UMAP on {len(sub)} patches …")
    reducer = umap.UMAP(n_neighbors=UMAP_NN, min_dist=UMAP_MDIST,
                        random_state=42, n_jobs=1)
    emb = reducer.fit_transform(X)

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(8, 7), facecolor="white")
    ax.set_facecolor("white")
    ax.spines[["top","right","left","bottom"]].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

    # background first
    bg_mask = sub["_is_bg"].values
    ax.scatter(emb[bg_mask, 0], emb[bg_mask, 1],
               c=C_BG, s=4, alpha=0.4, linewidths=0, rasterized=True)

    # test labeled (light colors, behind train)
    for label, color in [("adhesion", C_ADHE_TEST), ("No adhesion", C_NOAD_TEST)]:
        m = sub["_is_test"].values & (sub["_label"].values == label)
        ax.scatter(emb[m, 0], emb[m, 1],
                   c=color, s=18, alpha=0.85, linewidths=0, rasterized=True)

    # train labeled (dark colors, on top)
    for label, color in [("adhesion", C_ADHE_TRAIN), ("No adhesion", C_NOAD_TRAIN)]:
        m = sub["_is_train"].values & (sub["_label"].values == label)
        ax.scatter(emb[m, 0], emb[m, 1],
                   c=color, s=22, alpha=0.95, linewidths=0, rasterized=True)

    title_map = {"cp": "CellProfiler", "ilastik": "ilastik"}
    ax.set_title(f"{title_map[method]} features — DS1 B2 labels (fold {FOLD})",
                 fontsize=13, fontweight="bold", pad=10)

    legend_handles = [
        mpatches.Patch(color=C_ADHE_TRAIN, label="Adhesion (train)"),
        mpatches.Patch(color=C_NOAD_TRAIN, label="No adhesion (train)"),
        mpatches.Patch(color=C_ADHE_TEST,  label="Adhesion (test)"),
        mpatches.Patch(color=C_NOAD_TEST,  label="No adhesion (test)"),
        mpatches.Patch(color=C_BG,         label="Unlabeled (subsampled)"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, framealpha=0.9,
              loc="lower right", markerscale=1.5)

    n_tr  = int(sub["_is_train"].sum())
    n_te  = int(sub["_is_test"].sum())
    n_bg  = int(bg_mask.sum())
    ax.text(0.01, 0.01,
            f"train labels={n_tr}  test labels={n_te}  bg={n_bg}",
            transform=ax.transAxes, fontsize=8, color="#666666")

    out = OUT_DIR / f"umap_{method}_ds1.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for method, path in FEAT_FILES.items():
        make_umap(method, path)
    print("\nDone.")
