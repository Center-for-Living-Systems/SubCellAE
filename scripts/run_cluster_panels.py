#!/usr/bin/env python3
"""
KMeans clustering + center-patch panels for a trained AE model.

For each cluster, finds the N_PANEL patches closest to the cluster centroid
in latent space and saves a 4×4 panel as a grayscale float32 TIFF.

Output: <result_dir>/eval/cluster_panels/
  cluster_{k:02d}_n{total}.tif   — 4×4 panel, 16 patches nearest to centroid
  cluster_labels.csv             — per-patch cluster assignment + distance

Usage:
    python scripts/run_cluster_panels.py <result_dir> [--k 10] [--n-panel 16]
"""
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

N_PANEL  = 16   # patches per panel
COLS     = 4
ROWS     = 4
GAP      = 2    # px gap between patches
TITLE_H  = 22   # px for title bar per row


def _load_font(size: int = 13):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _title_bar(width: int, text: str, height: int = TITLE_H) -> np.ndarray:
    """Dark gray title bar with white text, returned as float32 (H, W)."""
    bar = np.full((height, width), 0.12, dtype=np.float32)
    pil = Image.fromarray((bar * 255).astype(np.uint8), mode="L")
    draw = ImageDraw.Draw(pil)
    draw.text((4, 3), text, fill=230, font=_load_font(13))
    return np.array(pil).astype(np.float32) / 255.0


def _scatter(emb: np.ndarray, labels, label_order: list, title: str,
             out_path: Path, cmap: str = "tab10"):
    fig, ax = plt.subplots(figsize=(7, 5))
    palette = plt.get_cmap(cmap)
    n = max(len(label_order) - 1, 1)
    for i, cat in enumerate(label_order):
        mask = np.array(labels) == cat
        if not mask.any():
            continue
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   label=str(cat), s=3, alpha=0.5,
                   color=palette(i / n))
    ax.set_title(title, fontsize=10)
    ax.legend(markerscale=3, fontsize=7, loc="best", ncol=2)
    ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def _make_panel(patches: list[np.ndarray], title: str,
                cols: int = COLS, gap: int = GAP) -> np.ndarray:
    """
    Build a grayscale float32 panel from a list of patches.
    For 1-ch patches (H, W): standard 4×4 grid.
    For 2-ch patches (2, H, W): paxillin section on top, actin on bottom,
    each with its own label bar.
    """
    if not patches:
        return np.zeros((1, 1), dtype=np.float32)

    two_ch = patches[0].ndim == 3 and patches[0].shape[0] == 2

    ps   = patches[0].shape[-1]   # patch side (32)
    rows = (N_PANEL + cols - 1) // cols
    cell = ps + gap
    W    = cols * cell - gap
    H_grid = rows * cell - gap

    def _fill_grid(ch_idx):
        canvas = np.zeros((H_grid, W), dtype=np.float32)
        for idx, p in enumerate(patches[:N_PANEL]):
            r, c = idx // cols, idx % cols
            y0, x0 = r * cell, c * cell
            patch = p[ch_idx] if two_ch else p
            patch = patch.astype(np.float32)
            lo, hi = float(patch.min()), float(patch.max())
            canvas[y0:y0+ps, x0:x0+ps] = (patch - lo) / (hi - lo + 1e-8)
        return canvas

    title_bar = _title_bar(W, title)

    if two_ch:
        pax_bar = _title_bar(W, "Paxillin (ch0)", height=16)
        act_bar = _title_bar(W, "Actin (ch1)",    height=16)
        return np.concatenate(
            [title_bar, pax_bar, _fill_grid(0), act_bar, _fill_grid(1)],
            axis=0)
    else:
        return np.concatenate([title_bar, _fill_grid(0)], axis=0)


def run(result_dir: Path, k: int = 10, n_panel: int = N_PANEL):
    csv_path = result_dir / "latents.csv"
    if not csv_path.exists():
        sys.exit(f"latents.csv not found in {result_dir}")

    recon_dir   = result_dir / "recon"
    idx_path    = recon_dir / "patches_index.csv"
    raw_tif     = recon_dir / "patches_raw.tif"
    for p in [idx_path, raw_tif]:
        if not p.exists():
            sys.exit(f"Missing: {p}")

    out_dir = result_dir / "eval" / "cluster_panels"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading latents: {result_dir.name}", flush=True)
    df = pd.read_csv(csv_path)
    latent_cols = [c for c in df.columns if c.startswith("z_")]
    latents = df[latent_cols].values.astype(np.float32)
    df["_stem"] = df["filename"].apply(lambda x: Path(x).stem)

    # join with patches_index to get frame indices
    idx_df = pd.read_csv(idx_path)
    merged = df.merge(idx_df[["name", "frame"]], left_on="_stem", right_on="name", how="left")
    missing = merged["frame"].isna().sum()
    if missing > 0:
        print(f"  Warning: {missing} patches not matched in patches_index.csv")
    merged = merged.dropna(subset=["frame"]).reset_index(drop=True)
    latents = merged[[c for c in latent_cols]].values.astype(np.float32)

    print(f"  Patches matched: {len(merged)}  |  latent dim: {latents.shape[1]}", flush=True)
    print(f"  Loading patches_raw.tif …", flush=True)
    raw_stack = tifffile.imread(str(raw_tif))   # (N, [C,] H, W)

    # KMeans
    print(f"  Running KMeans k={k} …", flush=True)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(latents)
    centroids      = km.cluster_centers_   # (k, D)

    # per-patch distance to assigned centroid
    dists = np.linalg.norm(latents - centroids[cluster_labels], axis=1)

    merged["cluster"]  = cluster_labels
    merged["dist_to_centroid"] = dists
    merged.to_csv(out_dir / "cluster_labels.csv", index=False)
    print(f"  Saved cluster_labels.csv", flush=True)

    # panel per cluster
    all_panels = []
    panel_index = []

    for ki in range(k):
        mask   = cluster_labels == ki
        n_total = int(mask.sum())
        sub    = merged[mask].copy()
        sub    = sub.sort_values("dist_to_centroid")          # closest first
        chosen = sub.head(n_panel)

        patches = []
        for _, row in chosen.iterrows():
            frame = int(row["frame"])
            raw_p = raw_stack[frame]   # (H,W) or (C,H,W)
            patches.append(raw_p.astype(np.float32))

        title  = f"Cluster {ki:02d}  (N={n_total})  {n_panel} nearest to centroid"
        panel  = _make_panel(patches, title)

        fname  = out_dir / f"cluster_{ki:02d}_n{n_total}.tif"
        tifffile.imwrite(str(fname), panel)
        print(f"  cluster {ki:02d}: {n_total:5d} patches  → {fname.name}", flush=True)

        all_panels.append(panel)
        panel_index.append({"frame": ki, "cluster": ki, "n_patches": n_total})

    # also save as a single stacked TIFF (pad to same height)
    max_h = max(p.shape[0] for p in all_panels)
    max_w = max(p.shape[1] for p in all_panels)
    padded = []
    for p in all_panels:
        ph, pw = p.shape
        pad = np.zeros((max_h - ph, max_w), dtype=np.float32)
        padded.append(np.concatenate([p, pad], axis=0) if ph < max_h else p)
        if pw < max_w:
            padded[-1] = np.concatenate(
                [padded[-1], np.zeros((max_h, max_w - pw), dtype=np.float32)], axis=1)

    tifffile.imwrite(str(out_dir / "all_clusters.tif"), np.stack(padded, axis=0))
    pd.DataFrame(panel_index).to_csv(out_dir / "panel_index.csv", index=False)
    print(f"\n  all_clusters.tif  ({k} frames)  → {out_dir}", flush=True)

    # ── scatter plots coloured by cluster ────────────────────────────────
    cluster_str   = [str(c) for c in cluster_labels]
    cluster_order = [str(i) for i in range(k)]
    eval_dir = result_dir / "eval"

    for emb_name, label in [("umap_emb.npy", "UMAP"), ("phate_emb.npy", "PHATE")]:
        emb_path = eval_dir / emb_name
        if not emb_path.exists():
            print(f"  {emb_name} not found – skipping {label} scatter", flush=True)
            continue
        emb = np.load(str(emb_path))
        if len(emb) != len(cluster_labels):
            print(f"  {label} embedding length {len(emb)} ≠ {len(cluster_labels)} patches – skipping")
            continue
        out_path = out_dir / f"{label.lower()}_kmeans_k{k}.png"
        _scatter(emb, cluster_str, cluster_order,
                 f"{label} – KMeans k={k}  ({result_dir.name})",
                 out_path)
        print(f"  {out_path.name}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--k",       type=int, default=10,
                        help="Number of KMeans clusters (default 10)")
    parser.add_argument("--n-panel", type=int, default=N_PANEL,
                        help="Patches per panel (default 16)")
    args = parser.parse_args()
    run(args.result_dir, k=args.k, n_panel=args.n_panel)


if __name__ == "__main__":
    main()
