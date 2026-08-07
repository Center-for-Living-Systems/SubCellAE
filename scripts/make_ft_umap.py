#!/usr/bin/env python3
"""
make_ft_umap.py

Generate UMAP comparison figures for vinc_control before and after
the Margaret fine-tuning:

  Left panel:  orig AE latents  (annabel_vinc_supcon2_s2v2)
  Right panel: FT AE latents   (annabel_vinc_margaret_ft_labeled_s2v2)

Each panel:
  - All 14 879 patches as small gray dots (background)
  - Labeled patches (Margaret 2-class) highlighted with GT colours:
      green = adhesion,  purple = No adhesion
  - LightGBM prediction boundary indicated by overall dot colour for unlabeled

Output:
  ft_comparison/ft_umap_margaret_labels.png   — GT coloring (2 panels)
  ft_comparison/ft_umap_predictions.png       — prediction coloring (2 panels)

Usage:
  python scripts/make_ft_umap.py [--split s2v2]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

REPO_ROOT  = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DATA_ROOT  = Path("/net/projects/CLS/lding/data/fa_data_analysis")
RUN_DIR    = DATA_ROOT / "ae_results" / "contrastive_run"
LABEL_DIR  = DATA_ROOT / "labelling"
PATCH_BASE = DATA_ROOT / "ae_results" / "patches" / "cio/vinc/control/tiff_patches32_mr10"

# GT colours (matching overlay script)
C_AD = np.array([0.13, 0.63, 0.24])   # green  — adhesion
C_NA = np.array([0.55, 0.18, 0.72])   # purple — No adhesion
C_BG = np.array([0.82, 0.82, 0.82])   # light gray — unlabeled background

LABEL_ORDER = ["No adhesion", "adhesion"]


def _compute_ft_latents(ft_ae, patch_dir: Path, device) -> pd.DataFrame:
    """Run FT AE encoder on all patches and return z_* DataFrame."""
    import torch, tifffile, re

    patches = sorted(patch_dir.glob("*.tif"))
    print(f"  Running FT AE on {len(patches)} patches…")
    records, imgs = [], []
    for p in patches:
        m = re.search(r"f(\d+)x(\d+)y(\d+)ps(\d+)", p.name)
        if not m:
            continue
        try:
            img = tifffile.imread(str(p)).astype(np.float32)
        except Exception:
            continue
        if img.ndim == 3:
            img = img[0]
        imgs.append(img)
        records.append({"filename": p.name})

    all_z = []
    ft_ae.eval()
    with torch.no_grad():
        for i in range(0, len(imgs), 512):
            batch = np.stack(imgs[i: i + 512])[:, None]
            t = torch.from_numpy(batch).to(device)
            _, z = ft_ae(t)
            all_z.append(z.cpu().numpy())

    Z = np.concatenate(all_z, axis=0)
    df = pd.DataFrame(records)
    for j in range(Z.shape[1]):
        df[f"z_{j}"] = Z[:, j]
    print(f"  Done: {len(df)} latents, dim={Z.shape[1]}")
    return df


def _fit_umap(Z: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1,
              random_state: int = 42) -> np.ndarray:
    from umap import UMAP
    reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                   random_state=random_state, n_jobs=1)
    return reducer.fit_transform(Z)


def _draw_umap_panel(ax, emb: np.ndarray, gt_map: dict, filenames: list,
                     pred_labels: list | None, title: str, show_gt: bool = True):
    """Draw one UMAP panel.

    If show_gt=True: gray BG dots + GT-colored labeled patches.
    If show_gt=False (prediction mode): all dots colored by prediction.
    """
    label_to_idx = {fn: i for i, fn in enumerate(filenames)}
    gt_idx   = {label_to_idx[fn]: lbl for fn, lbl in gt_map.items() if fn in label_to_idx}
    labeled  = sorted(gt_idx.keys())
    unlabeled = [i for i in range(len(filenames)) if i not in gt_idx]

    if show_gt:
        # background (unlabeled) — small gray dots
        ax.scatter(emb[unlabeled, 0], emb[unlabeled, 1],
                   c=[C_BG], s=2, alpha=0.3, linewidths=0, rasterized=True)
        # labeled patches — colored by GT
        colors = [C_AD if gt_idx[i] == "adhesion" else C_NA for i in labeled]
        ax.scatter(emb[labeled, 0], emb[labeled, 1],
                   c=colors, s=14, alpha=0.85, linewidths=0.2,
                   edgecolors="white", zorder=3, rasterized=True)
        legend_handles = [
            mpatches.Patch(color=C_AD, label=f"adhesion  (n={sum(1 for v in gt_idx.values() if v=='adhesion')})"),
            mpatches.Patch(color=C_NA, label=f"No adhesion  (n={sum(1 for v in gt_idx.values() if v=='No adhesion')})"),
            mpatches.Patch(color=C_BG, label=f"unlabeled  (n={len(unlabeled)})"),
        ]
    else:
        # all dots colored by prediction
        assert pred_labels is not None
        colors = [C_AD if p == "adhesion" else C_NA for p in pred_labels]
        ax.scatter(emb[:, 0], emb[:, 1],
                   c=colors, s=3, alpha=0.45, linewidths=0, rasterized=True)
        n_ad = sum(1 for p in pred_labels if p == "adhesion")
        n_na = len(pred_labels) - n_ad
        legend_handles = [
            mpatches.Patch(color=C_AD, label=f"adhesion  (n={n_ad})"),
            mpatches.Patch(color=C_NA, label=f"No adhesion  (n={n_na})"),
        ]

    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xlabel("UMAP 1", fontsize=8); ax.set_ylabel("UMAP 2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(handles=legend_handles, loc="best", fontsize=7, framealpha=0.8,
              markerscale=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="s2v2")
    args = ap.parse_args()

    orig_dir  = RUN_DIR / f"annabel_vinc_supcon2_{args.split}"
    ft_dir    = RUN_DIR / f"annabel_vinc_margaret_ft_labeled_{args.split}"
    out_dir   = orig_dir / "fa_cls_zrecon" / "ft_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load orig latents ────────────────────────────────────────────────────
    print("Loading orig AE latents…")
    orig_lat = pd.read_csv(orig_dir / "blind_test" / "vinc_control_latents.csv")
    z_cols   = [c for c in orig_lat.columns if c.startswith("z_")]
    filenames = orig_lat["filename"].tolist()
    Z_orig    = orig_lat[z_cols].values.astype(np.float32)

    # ── Compute FT latents ───────────────────────────────────────────────────
    ft_lat_cache = out_dir / "vinc_control_ft_latents.csv"
    if ft_lat_cache.exists():
        print("Loading FT latents from cache…")
        ft_lat = pd.read_csv(ft_lat_cache)
        # Align order to orig_lat filenames
        fn_to_row = ft_lat.set_index("filename")
        Z_ft = np.stack([fn_to_row.loc[fn, z_cols].values.astype(np.float32)
                         for fn in filenames])
    else:
        import torch
        device = torch.device("cpu")
        saved  = torch.load(str(ft_dir / "model_best.pt"), map_location=device, weights_only=False)
        ft_ae  = saved.to(device) if hasattr(saved, "forward") else None
        if ft_ae is None:
            from subcellae.modelling.autoencoders import ContrastiveAE
            ft_ae = ContrastiveAE(latent_dim=12, proj_dim=8, input_ps=32, no_ch=1).to(device)
            ft_ae.load_state_dict(saved)
        df_ft = _compute_ft_latents(ft_ae, PATCH_BASE, device)
        df_ft.to_csv(ft_lat_cache, index=False)
        print(f"  Cached → {ft_lat_cache.name}")
        fn_to_row = df_ft.set_index("filename")
        ft_z_cols = [c for c in df_ft.columns if c.startswith("z_")]
        Z_ft = np.stack([fn_to_row.loc[fn, ft_z_cols].values.astype(np.float32)
                         for fn in filenames])

    # ── Load Margaret's GT labels ────────────────────────────────────────────
    margaret = pd.read_csv(LABEL_DIR / "vinc_control_label_Margaret_2cls.csv")
    margaret["filename"] = margaret["unique_ID"].str.replace("-", "_", 1)
    gt_map = dict(zip(margaret["filename"], margaret["label"]))
    print(f"GT labels: {len(gt_map)} patches  "
          f"({sum(1 for v in gt_map.values() if v=='adhesion')} adh, "
          f"{sum(1 for v in gt_map.values() if v=='No adhesion')} no-adh)")

    # ── Load LightGBM models for prediction coloring ─────────────────────────
    import joblib, warnings
    warnings.filterwarnings("ignore")
    lgbm_orig = joblib.load(str(orig_dir / "fa_cls_zrecon" / "model.pkl"))
    lgbm_ft   = joblib.load(str(ft_dir / "lgbm_ft.pkl"))

    def _predict(Z, clf):
        return ["adhesion" if p == 1 else "No adhesion" for p in clf.predict(Z)]

    pred_orig = _predict(Z_orig, lgbm_orig)
    pred_ft   = _predict(Z_ft,   lgbm_ft)

    # ── Fit UMAP ─────────────────────────────────────────────────────────────
    umap_cache_orig = out_dir / "vinc_control_umap_orig.npy"
    umap_cache_ft   = out_dir / "vinc_control_umap_ft.npy"

    if umap_cache_orig.exists():
        print("Loading UMAP embeddings from cache…")
        emb_orig = np.load(str(umap_cache_orig))
        emb_ft   = np.load(str(umap_cache_ft))
    else:
        print("Fitting UMAP on orig latents…")
        emb_orig = _fit_umap(Z_orig)
        np.save(str(umap_cache_orig), emb_orig)
        print("Fitting UMAP on FT latents…")
        emb_ft = _fit_umap(Z_ft)
        np.save(str(umap_cache_ft), emb_ft)

    # ── Figure 1: GT label coloring ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor="white")
    _draw_umap_panel(axes[0], emb_orig, gt_map, filenames, None,
                     "Before FT — orig AE  (dataset1-only training)\nMargaret GT labels",
                     show_gt=True)
    _draw_umap_panel(axes[1], emb_ft, gt_map, filenames, None,
                     "After FT — AE fine-tuned on dataset1+Margaret\nMargaret GT labels",
                     show_gt=True)
    fig.suptitle("UMAP · dataset1/control · Margaret 2-class GT labels  "
                 "(green=adhesion  purple=No adhesion  gray=unlabeled)",
                 fontsize=9, y=0.02, color="gray")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out1 = out_dir / "ft_umap_margaret_labels.png"
    fig.savefig(str(out1), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out1.name}")

    # ── Figure 2: Prediction coloring ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor="white")
    _draw_umap_panel(axes[0], emb_orig, gt_map, filenames, pred_orig,
                     "Before FT — orig AE  (dataset1-only training)\nLightGBM predictions",
                     show_gt=False)
    _draw_umap_panel(axes[1], emb_ft, gt_map, filenames, pred_ft,
                     "After FT — AE fine-tuned on dataset1+Margaret\nLightGBM predictions",
                     show_gt=False)
    fig.suptitle("UMAP · dataset1/control · LightGBM 2-class predictions  "
                 "(green=adhesion  purple=No adhesion)",
                 fontsize=9, y=0.02, color="gray")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out2 = out_dir / "ft_umap_predictions.png"
    fig.savefig(str(out2), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out2.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
