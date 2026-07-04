#!/usr/bin/env python3
"""
regen_zproj_analysis.py

For each model: fit UMAP and KMeans on z_proj (p_ columns in latents.csv),
then generate cluster panels and scatter PNGs.

For baseline models without a projector (no p_ columns), falls back to z_.

Outputs per model:
  eval/umap_proj_emb.npy
  eval/umap_proj_annotation.png
  eval/umap_proj_condition.png
  eval/umap_proj_split.png
  eval/cluster_panels_proj/
    cluster_labels.csv
    cluster_XX_nNNN.tif
    all_clusters.tif
    umap_proj_kmeans_k10.png

Usage:
    python scripts/regen_zproj_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
from umap import UMAP

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from subcellae.utils.label_colors import (
    classification_label_to_color as FA_COLORS,
    condition_label_to_color      as CONDITION_COLORS,
    split_label_to_color          as SPLIT_COLORS,
    classification_label_order,
)

# ── config ─────────────────────────────────────────────────────────────────────

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

K          = 10
N_PANEL    = 16
COLS       = 4
GAP        = 2
TITLE_H    = 22


# ── scatter helpers ────────────────────────────────────────────────────────────

def _categorical_scatter(ax, x, y, labels, order, title, xlabel, ylabel,
                         explicit_colors=None):
    tab10 = plt.cm.tab10.colors
    for i, cat in enumerate(order):
        mask = np.array(labels) == cat
        if not mask.any():
            continue
        color = (explicit_colors[cat] if explicit_colors and cat in explicit_colors
                 else tab10[i % 10])
        ax.scatter(x[mask], y[mask], label=str(cat), s=4, alpha=0.6, color=color)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.legend(markerscale=3, fontsize=7, loc="best")


def _save_scatter(emb, df, col, order, title, save_path, explicit_colors=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    _categorical_scatter(ax, emb[:, 0], emb[:, 1], df[col].values,
                         order, title, "UMAP 1", "UMAP 2",
                         explicit_colors=explicit_colors)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


# ── annotation merge ───────────────────────────────────────────────────────────

def _merge_annotations(df: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Merge FA annotation from labels CSV via filename+condition."""
    df = df.copy()
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


# ── cluster panel helpers ──────────────────────────────────────────────────────

def _load_font(size: int = 13):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _title_bar(width: int, text: str, height: int = TITLE_H) -> np.ndarray:
    bar = np.full((height, width), 0.12, dtype=np.float32)
    pil = Image.fromarray((bar * 255).astype(np.uint8), mode="L")
    draw = ImageDraw.Draw(pil)
    draw.text((4, 3), text, fill=230, font=_load_font(13))
    return np.array(pil).astype(np.float32) / 255.0


def _make_panel(patches: list, title: str) -> np.ndarray:
    if not patches:
        return np.zeros((1, 1), dtype=np.float32)
    two_ch = patches[0].ndim == 3 and patches[0].shape[0] == 2
    ps     = patches[0].shape[-1]
    rows   = (N_PANEL + COLS - 1) // COLS
    cell   = ps + GAP
    W      = COLS * cell - GAP
    H_grid = rows * cell - GAP

    def _fill_grid(ch_idx):
        canvas = np.zeros((H_grid, W), dtype=np.float32)
        for idx, p in enumerate(patches[:N_PANEL]):
            r, c = idx // COLS, idx % COLS
            y0, x0 = r * cell, c * cell
            patch = (p[ch_idx] if two_ch else p).astype(np.float32)
            lo, hi = float(patch.min()), float(patch.max())
            canvas[y0:y0+ps, x0:x0+ps] = (patch - lo) / (hi - lo + 1e-8)
        return canvas

    tb = _title_bar(W, title)
    if two_ch:
        return np.concatenate(
            [tb, _title_bar(W, "Paxillin (ch0)", 16), _fill_grid(0),
             _title_bar(W, "Actin (ch1)", 16),    _fill_grid(1)], axis=0)
    return np.concatenate([tb, _fill_grid(0)], axis=0)


# ── per-model ──────────────────────────────────────────────────────────────────

def _process_model(model_dir: Path, labels: pd.DataFrame):
    csv_path = model_dir / "latents.csv"
    if not csv_path.exists():
        print(f"  SKIP {model_dir.name}: no latents.csv")
        return

    recon_dir = model_dir / "recon"
    raw_tif   = recon_dir / "patches_raw.tif"
    idx_path  = recon_dir / "patches_index.csv"
    if not raw_tif.exists() or not idx_path.exists():
        print(f"  SKIP {model_dir.name}: missing recon TIFs")
        return

    eval_dir  = model_dir / "eval"
    out_dir   = eval_dir / "cluster_panels_proj"
    eval_dir.mkdir(exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  {model_dir.name} …", flush=True)
    df = pd.read_csv(csv_path)
    df = _merge_annotations(df, labels)

    # choose projection space
    proj_cols   = [c for c in df.columns if c.startswith("p_")]
    latent_cols = proj_cols if proj_cols else [c for c in df.columns if c.startswith("z_")]
    space_label = "z_proj" if proj_cols else "z_recon (no projector)"
    print(f"    using {space_label}  ({len(latent_cols)} dims)", flush=True)

    latents = df[latent_cols].values.astype(np.float32)

    # ── UMAP ────────────────────────────────────────────────────────────────
    print(f"    UMAP …", flush=True)
    reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    emb = reducer.fit_transform(latents)
    np.save(str(eval_dir / "umap_proj_emb.npy"), emb)

    # annotation scatter (annotated rows only)
    df_ann  = df[df["annotation_label_name"].notna()].copy()
    emb_ann = emb[df["annotation_label_name"].notna().values]
    if len(df_ann):
        _save_scatter(emb_ann, df_ann, "annotation_label_name", FA_ORDER,
                      "FA annotation (z_proj)",
                      eval_dir / "umap_proj_annotation.png",
                      FA_COLORS)

    # condition scatter
    cond_order = [c for c in COND_ORDER if c in df["condition_name"].values]
    _save_scatter(emb, df, "condition_name", cond_order,
                  "Condition (z_proj)",
                  eval_dir / "umap_proj_condition.png",
                  CONDITION_COLORS)

    # split scatter
    split_order = [s for s in SPLIT_ORDER if s in df["split"].values]
    _save_scatter(emb, df, "split", split_order,
                  "Split (z_proj)",
                  eval_dir / "umap_proj_split.png",
                  SPLIT_COLORS)

    # ── KMeans ──────────────────────────────────────────────────────────────
    print(f"    KMeans k={K} …", flush=True)

    # align df with patches_index via stem match
    idx_df = pd.read_csv(idx_path)
    df["_stem"] = df["filename"].apply(lambda x: Path(x).stem)
    merged = df.merge(idx_df[["name", "frame"]], left_on="_stem", right_on="name", how="left")
    merged = merged.dropna(subset=["frame"]).reset_index(drop=True)
    km_latents = merged[latent_cols].values.astype(np.float32)

    km = KMeans(n_clusters=K, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(km_latents)
    centroids      = km.cluster_centers_
    dists          = np.linalg.norm(km_latents - centroids[cluster_labels], axis=1)
    merged["cluster"]          = cluster_labels
    merged["dist_to_centroid"] = dists
    merged.to_csv(out_dir / "cluster_labels.csv", index=False)

    # ── cluster panels ───────────────────────────────────────────────────────
    print(f"    Cluster panels …", flush=True)
    raw_stack  = tifffile.imread(str(raw_tif))
    all_panels = []
    panel_index = []

    for ki in range(K):
        mask    = cluster_labels == ki
        n_total = int(mask.sum())
        sub     = merged[mask].sort_values("dist_to_centroid").head(N_PANEL)
        patches = [raw_stack[int(row["frame"])].astype(np.float32)
                   for _, row in sub.iterrows()]
        title   = f"Cluster {ki:02d}  (N={n_total})  {N_PANEL} nearest to centroid"
        panel   = _make_panel(patches, title)
        fname   = out_dir / f"cluster_{ki:02d}_n{n_total}.tif"
        tifffile.imwrite(str(fname), panel)
        all_panels.append(panel)
        panel_index.append({"frame": ki, "cluster": ki, "n_patches": n_total})
        print(f"      cluster {ki:02d}: {n_total:5d} patches", flush=True)

    # stacked TIFF
    max_h = max(p.shape[0] for p in all_panels)
    max_w = max(p.shape[1] for p in all_panels)
    padded = []
    for p in all_panels:
        ph, pw = p.shape
        row = np.concatenate([p, np.zeros((max_h - ph, pw), dtype=np.float32)], axis=0) \
              if ph < max_h else p
        if pw < max_w:
            row = np.concatenate([row, np.zeros((max_h, max_w - pw), dtype=np.float32)], axis=1)
        padded.append(row)
    tifffile.imwrite(str(out_dir / "all_clusters.tif"), np.stack(padded, axis=0))
    pd.DataFrame(panel_index).to_csv(out_dir / "panel_index.csv", index=False)

    # UMAP scatter coloured by z_proj cluster (using full df embedding)
    # align cluster labels back to full df order via merged frame index
    frame_to_cluster = dict(zip(merged["frame"].astype(int), merged["cluster"]))
    full_cluster = np.full(len(emb), -1, dtype=int)
    for i, row in df.iterrows():
        stem = Path(row["filename"]).stem
        match = idx_df[idx_df["name"] == stem]
        if not match.empty:
            f = int(match.iloc[0]["frame"])
            full_cluster[i] = frame_to_cluster.get(f, -1)

    valid = full_cluster >= 0
    cluster_str   = [str(c) for c in full_cluster[valid]]
    cluster_order = [str(i) for i in range(K)]
    fig, ax = plt.subplots(figsize=(7, 6))
    tab10 = plt.cm.tab10.colors
    for i, cat in enumerate(cluster_order):
        mask = np.array(cluster_str) == cat
        if not mask.any():
            continue
        ax.scatter(emb[valid][mask, 0], emb[valid][mask, 1],
                   label=cat, s=4, alpha=0.6, color=tab10[i % 10])
    ax.set_title(f"UMAP z_proj – KMeans k={K}")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=3, fontsize=7, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(str(out_dir / f"umap_proj_kmeans_k{K}.png"), dpi=150)
    plt.close(fig)

    print(f"    Done → {out_dir}", flush=True)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    labels = pd.read_csv(LABEL_CSV)
    for model_name in MODELS:
        _process_model(RUNS / model_name, labels)
    print("\nAll done.")


if __name__ == "__main__":
    main()
