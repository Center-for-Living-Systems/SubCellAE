#!/usr/bin/env python3
"""
regen_scatter_plots.py

Regenerate UMAP scatter PNGs (annotation / condition / split) for a list of
models without re-fitting the UMAP embedding.

Steps per model:
  1. Load saved umap_emb.npy
  2. Load latents.csv and merge annotation labels from the labels CSV
  3. Re-render umap_annotation.png, umap_condition.png, umap_split.png

Usage:
    python scripts/regen_scatter_plots.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import _categorical_scatter directly to avoid czifile dependency in analysis_pipeline
def _categorical_scatter(ax, x, y, labels, order, title, xlabel, ylabel,
                         cmap="tab10", explicit_colors=None):
    tab10 = plt.cm.tab10.colors
    for i, cat in enumerate(order):
        mask = np.array(labels) == cat
        color = (explicit_colors[cat] if explicit_colors and cat in explicit_colors
                 else tab10[i % 10])
        ax.scatter(x[mask], y[mask], label=str(cat), s=4, alpha=0.6, color=color)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.legend(markerscale=3, fontsize=7, loc="best")

from subcellae.utils.label_colors import (
    classification_label_to_color as FA_COLORS,
    condition_label_to_color      as CONDITION_COLORS,
    split_label_to_color          as SPLIT_COLORS,
    classification_label_order,
    position_label_order,
)

# ── config ────────────────────────────────────────────────────────────────────

RUNS      = Path("/net/projects/CLS/lding/data/fa_data_analysis/ae_results/contrastive_run")
LABEL_CSV = Path("/net/projects/CLS/lding/data/fa_data_analysis/labelling/labels_vinc_20260521.csv")

FA_ORDER    = [l for l in classification_label_order if l != "Uncertain"]
COND_ORDER  = list(CONDITION_COLORS.keys())
SPLIT_ORDER = list(SPLIT_COLORS.keys())

MODELS = [
    "baseline_vinc_only_ch3",
    "contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_ch3",
    "contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_lc025_ch3",
    "contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1",
    "contrastive_cio_rb_vinc_lat12proj8_enlcrop_sc2_nl1_lc025",
    "baseline_vinc_2ch_pax_act",
    "contrastive_cio_rb_vinc_lat12proj8_sc2_nl1_2ch_pax_act",
    "contrastive_cio_rb_vinc_lat12proj8_sc2_nl1_lc025_2ch_pax_act",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _save_scatter(emb, df, col, order, title, save_path, explicit_colors):
    fig, ax = plt.subplots(figsize=(7, 6))
    _categorical_scatter(
        ax,
        emb[:, 0], emb[:, 1],
        df[col].values,
        order,
        title=title,
        xlabel="UMAP 1",
        ylabel="UMAP 2",
        explicit_colors=explicit_colors,
    )
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


def _load_and_merge(model_dir: Path, labels: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(model_dir / "latents.csv")
    # strip 'control_' / 'ycomp_' prefix to match label filenames
    df["_bare"] = df["filename"].str.replace(r"^[^_]+_", "", regex=True)
    merged = df.merge(
        labels[["crop_img_filename", "condition", "classification"]],
        left_on=["_bare", "condition_name"],
        right_on=["crop_img_filename", "condition"],
        how="left",
    )
    merged.drop(columns=["_bare", "crop_img_filename", "condition"], errors="ignore",
                inplace=True)
    merged["annotation_label_name"] = merged["classification"].where(
        merged["classification"].isin(FA_ORDER), other=np.nan
    )
    return merged


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    labels = pd.read_csv(LABEL_CSV)

    for model_name in MODELS:
        model_dir = RUNS / model_name
        eval_dir  = model_dir / "eval"
        emb_path  = eval_dir / "umap_emb.npy"

        if not emb_path.exists():
            print(f"  SKIP {model_name}: no umap_emb.npy")
            continue

        print(f"  {model_name} …", flush=True)
        emb = np.load(str(emb_path))
        df  = _load_and_merge(model_dir, labels)

        if len(df) != len(emb):
            print(f"    WARNING: latents ({len(df)}) != umap ({len(emb)}); skipping")
            continue

        n_ann = df["annotation_label_name"].notna().sum()
        print(f"    annotated patches: {n_ann} / {len(df)}")

        # ── annotation scatter ──
        df_ann = df[df["annotation_label_name"].notna()].copy()
        emb_ann = emb[df["annotation_label_name"].notna().values]
        if len(df_ann):
            _save_scatter(emb_ann, df_ann, "annotation_label_name", FA_ORDER,
                          "FA annotation", eval_dir / "umap_annotation.png",
                          FA_COLORS)

        # ── condition scatter ──
        cond_order = [c for c in COND_ORDER if c in df["condition_name"].values]
        _save_scatter(emb, df, "condition_name", cond_order,
                      "Condition", eval_dir / "umap_condition.png",
                      CONDITION_COLORS)

        # ── split scatter ──
        split_order = [s for s in SPLIT_ORDER if s in df["split"].values]
        _save_scatter(emb, df, "split", split_order,
                      "Split", eval_dir / "umap_split.png",
                      SPLIT_COLORS)

    print("\nDone.")


if __name__ == "__main__":
    main()
