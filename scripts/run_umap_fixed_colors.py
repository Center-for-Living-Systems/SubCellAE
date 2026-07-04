#!/usr/bin/env python3
"""
run_umap_fixed_colors.py

Redraw UMAP / PHATE scatter plots with fixed, consistent color palettes.
Loads existing umap_emb.npy / phate_emb.npy — no re-embedding.

Plots produced (saved to <result_dir>/eval/):
  umap_annotation.png      FA type, fixed FA_COLORS
  phate_annotation.png     FA type, fixed FA_COLORS
  umap_position.png        Cell position, fixed POS_COLORS  (new)
  phate_position.png       Cell position, fixed POS_COLORS  (new)
  umap_ppax_true.png       ppax true FA labels, FA_COLORS  (if ppax CSV exists)
  umap_ppax_pred.png       ppax predicted FA labels, FA_COLORS  (if ppax CSV exists)

Usage:
    python scripts/run_umap_fixed_colors.py <result_dir> [<result_dir2> ...]
    python scripts/run_umap_fixed_colors.py --all    # all models in contrastive_run
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT        = "/net/projects/CLS/lding/data/fa_data_analysis"
RUNS        = f"{ROOT}/ae_results/contrastive_run"
VINC_LABELS = f"{ROOT}/labelling/labels_vinc_20260521.csv"
PPAX_LABELS = f"{ROOT}/labelling/labels_ppax_20260521.csv"

FA_ORDER  = [
    "Nascent Adhesion", "focal complex", "focal adhesion",
    "fibrillar adhesion", "No adhesion",
]
FA_COLORS = ["#e6194b", "#f58231", "#3cb44b", "#4363d8", "#aaaaaa"]

POS_ORDER  = [
    "Cell Protruding Edge", "Cell Periphery/other", "Lamella", "Cell Body",
]
POS_COLORS = ["#e6194b", "#f58231", "#3cb44b", "#4363d8"]

FA_COLOR_MAP  = dict(zip(FA_ORDER,  FA_COLORS))
POS_COLOR_MAP = dict(zip(POS_ORDER, POS_COLORS))
UNLABELLED_COLOR = "#cccccc"

FIG_SIZE   = (7, 5)
POINT_SIZE = 3
ALPHA      = 0.5


# ── plotting ──────────────────────────────────────────────────────────────────

def _scatter_fixed(emb: np.ndarray, labels: np.ndarray,
                   order: list, color_map: dict,
                   title: str, out_path: Path) -> None:
    """Scatter with fixed per-category colors; unlabelled points plotted last in gray."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # plot each category in order so legend is consistent
    for cat in order:
        mask = labels == cat
        if not mask.any():
            continue
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=color_map[cat], s=POINT_SIZE, alpha=ALPHA,
                   linewidths=0, label=cat, rasterized=True)

    # unlabelled / unknown
    unknown_mask = ~np.isin(labels, order)
    if unknown_mask.any():
        ax.scatter(emb[unknown_mask, 0], emb[unknown_mask, 1],
                   c=UNLABELLED_COLOR, s=POINT_SIZE * 0.6, alpha=0.3,
                   linewidths=0, label="unlabelled", rasterized=True)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")

    # legend with larger markers
    handles = [mpatches.Patch(color=color_map[c], label=c)
               for c in order if np.any(labels == c)]
    if unknown_mask.any():
        handles.append(mpatches.Patch(color=UNLABELLED_COLOR, label="unlabelled"))
    ax.legend(handles=handles, fontsize=7, markerscale=1,
              loc="best", framealpha=0.7)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"  → {out_path.name}", flush=True)


# ── label loading ─────────────────────────────────────────────────────────────

def _load_vinc_labels() -> pd.DataFrame | None:
    if not Path(VINC_LABELS).exists():
        return None
    df = pd.read_csv(VINC_LABELS)
    df["unique_ID"] = df["unique_ID"].apply(lambda p: Path(p).name)
    return df      # cols: unique_ID, classification, Position


def _match_labels(latents_df: pd.DataFrame,
                  label_df: pd.DataFrame) -> pd.DataFrame:
    """Merge FA classification + Position into latents_df by filename."""
    latents_df = latents_df.copy()
    latents_df["_uid"] = latents_df["filename"].apply(
        lambda f: Path(f).name.replace("_", "-", 1))
    uid_to_fa  = dict(zip(label_df["unique_ID"], label_df["classification"].astype(str)))
    uid_to_pos = dict(zip(label_df["unique_ID"], label_df["Position"].astype(str)))
    latents_df["fa_label"]  = latents_df["_uid"].map(uid_to_fa)
    latents_df["pos_label"] = latents_df["_uid"].map(uid_to_pos)
    return latents_df


# ── per-model runner ──────────────────────────────────────────────────────────

def run_one(result_dir: Path, label_df: pd.DataFrame | None) -> None:
    eval_dir = result_dir / "eval"
    umap_path  = eval_dir / "umap_emb.npy"
    phate_path = eval_dir / "phate_emb.npy"

    if not umap_path.exists():
        print(f"  [skip] no umap_emb.npy in {result_dir.name}", flush=True)
        return

    print(f"\n{'='*60}", flush=True)
    print(f"{result_dir.name}", flush=True)

    umap_emb  = np.load(str(umap_path))
    phate_emb = np.load(str(phate_path)) if phate_path.exists() else None

    # load latents for filename → label matching
    lat_csv = result_dir / "latents.csv"
    if not lat_csv.exists():
        print("  [skip] no latents.csv", flush=True)
        return
    lat_df = pd.read_csv(lat_csv, low_memory=False)

    # ── FA + Position labels ──────────────────────────────────────────────────
    if label_df is not None:
        lat_df = _match_labels(lat_df, label_df)
        fa_labels  = lat_df["fa_label"].fillna("unlabelled").values
        pos_labels = lat_df["pos_label"].fillna("unlabelled").values
        has_fa  = np.isin(fa_labels,  FA_ORDER).any()
        has_pos = np.isin(pos_labels, POS_ORDER).any()
    else:
        has_fa = has_pos = False

    n_emb = len(umap_emb)

    if has_fa:
        fa_emb_labels = fa_labels[:n_emb]
        _scatter_fixed(umap_emb, fa_emb_labels, FA_ORDER, FA_COLOR_MAP,
                       f"UMAP – FA type  ({result_dir.name})",
                       eval_dir / "umap_annotation.png")
        if phate_emb is not None and len(phate_emb) == n_emb:
            _scatter_fixed(phate_emb, fa_emb_labels, FA_ORDER, FA_COLOR_MAP,
                           f"PHATE – FA type  ({result_dir.name})",
                           eval_dir / "phate_annotation.png")
    else:
        print("  no FA labels matched — skipping umap_annotation / phate_annotation",
              flush=True)

    if has_pos:
        pos_emb_labels = pos_labels[:n_emb]
        _scatter_fixed(umap_emb, pos_emb_labels, POS_ORDER, POS_COLOR_MAP,
                       f"UMAP – Cell position  ({result_dir.name})",
                       eval_dir / "umap_position.png")
        if phate_emb is not None and len(phate_emb) == n_emb:
            _scatter_fixed(phate_emb, pos_emb_labels, POS_ORDER, POS_COLOR_MAP,
                           f"PHATE – Cell position  ({result_dir.name})",
                           eval_dir / "phate_position.png")
    else:
        print("  no position labels matched — skipping umap_position / phate_position",
              flush=True)

    # ── ppax transfer scatter (true + pred) ───────────────────────────────────
    ppax_csv = eval_dir / "ppax_transfer.csv"
    if ppax_csv.exists():
        ppax_df = pd.read_csv(ppax_csv, low_memory=False)
        ppax_umap_path = eval_dir / "umap_ppax_emb.npy"
        if ppax_umap_path.exists():
            ppax_umap = np.load(str(ppax_umap_path))
            if "true_label" in ppax_df.columns:
                true_labels = ppax_df["true_label"].fillna("unlabelled").values
                labelled_mask = np.isin(true_labels, FA_ORDER)
                if labelled_mask.any():
                    _scatter_fixed(ppax_umap[labelled_mask],
                                   true_labels[labelled_mask],
                                   FA_ORDER, FA_COLOR_MAP,
                                   f"ppax UMAP – true FA label",
                                   eval_dir / "umap_ppax_true.png")
            if "pred_label" in ppax_df.columns:
                pred_labels = ppax_df["pred_label"].fillna("unlabelled").values
                labelled_mask = np.isin(true_labels if "true_label" in ppax_df.columns
                                        else pred_labels, FA_ORDER)
                if labelled_mask.any():
                    _scatter_fixed(ppax_umap[labelled_mask],
                                   pred_labels[labelled_mask],
                                   FA_ORDER, FA_COLOR_MAP,
                                   f"ppax UMAP – predicted FA label",
                                   eval_dir / "umap_ppax_pred.png")
        else:
            # ppax umap was embedded onto existing UMAP space: look for pts in latents
            # try reading existing ppax_true/pred PNGs which were in vinc umap space
            # (these exist as umap_ppax_true.png already in eval_dir — already handled above)
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dirs", nargs="*", type=Path)
    parser.add_argument("--all", action="store_true",
                        help="Run on all model dirs under contrastive_run/")
    args = parser.parse_args()

    if args.all:
        dirs = sorted(Path(RUNS).iterdir())
        dirs = [d for d in dirs if d.is_dir() and (d / "eval" / "umap_emb.npy").exists()]
    else:
        dirs = args.result_dirs
        if not dirs:
            parser.print_help(); sys.exit(1)

    print(f"Loading vinc label CSV …", flush=True)
    label_df = _load_vinc_labels()
    if label_df is None:
        print("WARNING: vinc labels CSV not found — no FA/position scatter plots",
              flush=True)

    print(f"Processing {len(dirs)} model dir(s) …", flush=True)
    for d in dirs:
        if not d.is_dir():
            print(f"[skip] not a dir: {d}"); continue
        run_one(d, label_df)

    print("\nDone.")


if __name__ == "__main__":
    main()
