#!/usr/bin/env python3
"""
run_ds_combo_analysis.py

UMAP + KMeans cluster panels for a single ds-combo model dir.
Reads eval/latents.csv (generated during training), fits UMAP on z_proj
(or z_recon if no projector), generates:
  eval/umap_combo_condition.png
  eval/umap_combo_split.png
  eval/cluster_panels_combo/
    cluster_labels.csv
    cluster_XX_nNNN.tif
    all_clusters.tif
    umap_combo_kmeans_k10.png

Usage:
  python scripts/run_ds_combo_analysis.py <model_dir>
  python scripts/run_ds_combo_analysis.py <model_dir> --k 10 --n-panel 16
"""
from __future__ import annotations

import argparse
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

# ── palette ───────────────────────────────────────────────────────────────────

DS_COLORS = {
    "vinc":   "#4C72B0",
    "nih3t3": "#DD8452",
    "ppax":   "#55A868",
    "pfak":   "#C44E52",
}
COND_MARKER = {"control": "o", "ycomp": "^"}
SPLIT_COLOR = {"train": "#333333", "val": "#AAAAAA"}


def _ds_from_condition(cond_name: str) -> str:
    for ds in DS_COLORS:
        if cond_name.startswith(ds):
            return ds
    return "vinc"


def _cond_suffix(cond_name: str) -> str:
    return "ycomp" if "ycomp" in cond_name else "control"


# ── cluster panels ────────────────────────────────────────────────────────────

def _make_cluster_panels(labels: np.ndarray, patch_paths: list[Path],
                          out_dir: Path, k: int, n_panel: int,
                          umap_emb: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_tifs = []
    all_rows = []

    for c in range(k):
        idx = np.where(labels == c)[0]
        np.random.shuffle(idx)
        sel = idx[:n_panel]
        imgs = []
        for i in sel:
            p = patch_paths[i]
            if p.exists():
                img = tifffile.imread(str(p)).astype(np.float32)
                if img.ndim == 3:
                    img = img[0]
                mx = float(np.percentile(img, 99.9))
                if mx > 0:
                    img = np.clip(img, 0, mx) / mx
                imgs.append((img * 255).astype(np.uint8))
        if not imgs:
            continue

        ps = imgs[0].shape[0]
        cols = min(n_panel, len(imgs))
        panel = np.zeros((ps, ps * cols), dtype=np.uint8)
        for j, img in enumerate(imgs):
            panel[:, j*ps:(j+1)*ps] = img
        tif_name = out_dir / f"cluster_{c:02d}_n{len(idx)}.tif"
        tifffile.imwrite(str(tif_name), panel)
        panel_tifs.append(panel)
        all_rows.append({"cluster": c, "n_patches": len(idx), "file": tif_name.name})

    if panel_tifs:
        max_w = max(p.shape[1] for p in panel_tifs)
        stacked = []
        for p in panel_tifs:
            if p.shape[1] < max_w:
                pad = np.zeros((p.shape[0], max_w - p.shape[1]), dtype=np.uint8)
                p = np.concatenate([p, pad], axis=1)
            stacked.append(p)
        tifffile.imwrite(str(out_dir / "all_clusters.tif"), np.stack(stacked))

    pd.DataFrame(all_rows).to_csv(out_dir / "cluster_labels.csv", index=False)

    # KMeans scatter on UMAP
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(umap_emb[:, 0], umap_emb[:, 1], c=labels,
                    cmap="tab20", s=2, alpha=0.4, linewidths=0)
    plt.colorbar(sc, ax=ax, label="cluster")
    ax.set_title(f"KMeans k={k}")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    fig.tight_layout()
    fig.savefig(str(out_dir / f"umap_combo_kmeans_k{k}.png"), dpi=150)
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def run(model_dir: Path, k: int = 10, n_panel: int = 16) -> None:
    eval_dir   = model_dir / "eval"
    latents_csv = eval_dir / "latents.csv"
    if not latents_csv.exists():
        print(f"  SKIP: no latents.csv in {eval_dir}")
        return

    df = pd.read_csv(latents_csv)
    print(f"  {len(df)} patches, columns: {list(df.columns[:8])}…")

    # Choose feature space: prefer z_proj (p_ cols), fallback to z_ cols
    proj_cols  = [c for c in df.columns if c.startswith("p_")]
    recon_cols = [c for c in df.columns if c.startswith("z_")]
    feat_cols  = proj_cols if proj_cols else recon_cols
    feat_label = "z_proj" if proj_cols else "z_recon"
    print(f"  Using {feat_label} ({len(feat_cols)} dims)")

    Z = df[feat_cols].values.astype(np.float32)

    # UMAP
    print("  Fitting UMAP…")
    emb = UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1).fit_transform(Z)
    np.save(str(eval_dir / "umap_combo_emb.npy"), emb)

    # ── scatter: condition ────────────────────────────────────────────────────
    cond_col = "condition_name" if "condition_name" in df.columns else None
    fig, ax = plt.subplots(figsize=(7, 6))
    if cond_col:
        for cond_name, grp in df.groupby(cond_col):
            ds   = _ds_from_condition(str(cond_name))
            suf  = _cond_suffix(str(cond_name))
            color  = DS_COLORS.get(ds, "#888888")
            marker = COND_MARKER.get(suf, "o")
            ax.scatter(emb[grp.index, 0], emb[grp.index, 1],
                       c=color, marker=marker, s=2, alpha=0.4,
                       linewidths=0, label=cond_name)
        ax.legend(markerscale=4, fontsize=7, loc="best")
    else:
        ax.scatter(emb[:, 0], emb[:, 1], s=2, alpha=0.3)
    ax.set_title(f"UMAP {feat_label} — condition"); ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    fig.tight_layout()
    fig.savefig(str(eval_dir / "umap_combo_condition.png"), dpi=150)
    plt.close(fig)
    print("  Saved umap_combo_condition.png")

    # ── scatter: train/val split ──────────────────────────────────────────────
    if "split" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 6))
        for spl, grp in df.groupby("split"):
            ax.scatter(emb[grp.index, 0], emb[grp.index, 1],
                       c=SPLIT_COLOR.get(str(spl), "#888888"), s=2,
                       alpha=0.4, linewidths=0, label=str(spl))
        ax.legend(markerscale=4, fontsize=8)
        ax.set_title(f"UMAP {feat_label} — split"); ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        fig.tight_layout()
        fig.savefig(str(eval_dir / "umap_combo_split.png"), dpi=150)
        plt.close(fig)
        print("  Saved umap_combo_split.png")

    # ── KMeans + cluster panels ───────────────────────────────────────────────
    print(f"  KMeans k={k}…")
    km_labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(Z)

    patch_paths: list[Path] = []
    if "patch_path" in df.columns:
        patch_paths = [Path(p) for p in df["patch_path"]]
    elif "filename" in df.columns and "patch_dir" in df.columns:
        patch_paths = [Path(str(d)) / str(f) for d, f in zip(df["patch_dir"], df["filename"])]

    panels_dir = eval_dir / "cluster_panels_combo"
    if patch_paths:
        _make_cluster_panels(km_labels, patch_paths, panels_dir, k, n_panel, emb)
        print(f"  Cluster panels → {panels_dir}/")
    else:
        print("  WARNING: no patch paths in latents.csv — skipping cluster panels")
        panels_dir.mkdir(parents=True, exist_ok=True)
        # still save the kmeans scatter
        _make_cluster_panels.__globals__  # no-op, just save scatter
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(emb[:, 0], emb[:, 1], c=km_labels,
                   cmap="tab20", s=2, alpha=0.4, linewidths=0)
        ax.set_title(f"KMeans k={k}")
        fig.tight_layout()
        fig.savefig(str(panels_dir / f"umap_combo_kmeans_k{k}.png"), dpi=150)
        plt.close(fig)
        print(f"  KMeans scatter → {panels_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--k",       type=int, default=10, help="KMeans clusters")
    parser.add_argument("--n-panel", type=int, default=16, help="Patches per cluster panel")
    args = parser.parse_args()

    if not args.model_dir.is_dir():
        sys.exit(f"Not a directory: {args.model_dir}")

    print(f"Model dir: {args.model_dir}")
    run(args.model_dir, k=args.k, n_panel=args.n_panel)
    print("Done.")


if __name__ == "__main__":
    main()
