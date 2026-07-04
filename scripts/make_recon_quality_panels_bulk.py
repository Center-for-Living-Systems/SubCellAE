#!/usr/bin/env python3
"""
30-patch reconstruction quality panels (bulk version).

Two modes:

  vinc-unlabelled
    Uses existing recon TIFs in <variant_dir>/recon/, filtered to patches
    that have no FA-type annotation (annotation_label == -1 in latents.csv).
    Split (train/val) shown as patch title.

  external  (e.g. ppax)
    Loads <variant_dir>/model_final.pt, runs inference on all patches in
    --patch-dir, computes metrics, generates panels.
    Condition name shown as patch title.

Layout: 30 patches per panel, 6 cols × 5 row-pairs.
  top row of each pair    = raw
  bottom row of each pair = recon
  Bold "raw"/"recon" labels on left of each pair row.
  Empty slots left blank (axis off).

Output: <variant_dir>/quality_panels_bulk/{subset}_{metric}_pct{P}.png

Usage
-----
  # vinc unlabelled
  python scripts/make_recon_quality_panels_bulk.py <variant_dir> --subset unlabelled

  # ppax (runs inference)
  python scripts/make_recon_quality_panels_bulk.py <variant_dir> \\
      --subset ppax \\
      --patch-dirs /path/to/ppax/control/tiff_patches32 \\
                   /path/to/ppax/ycomp/tiff_patches32
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tifffile
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parents[1]))
from subcellae.modelling.dataset import PatchDataset, MultiChannelPatchDataset


N_PATCHES   = 30
COLS        = 6
PERCENTILES = list(range(10, 100, 10))
METRICS     = ["recon_l1", "recon_mse", "recon_hessian_l1"]

_METRIC_SLUG = {
    "recon_l1":         "l1",
    "recon_mse":        "mse",
    "recon_hessian_l1": "hessian",
}


# ── metric helpers ────────────────────────────────────────────────────────────

def _compute_l1(raw, recon):
    return np.abs(raw.astype(np.float32) - recon.astype(np.float32)).mean(
        axis=tuple(range(1, raw.ndim)))

def _compute_mse(raw, recon):
    d = raw.astype(np.float32) - recon.astype(np.float32)
    return (d * d).mean(axis=tuple(range(1, raw.ndim)))

def _compute_hessian_l1(raw, recon):
    if raw.ndim == 4:
        return np.mean([_compute_hessian_l1(raw[:, c], recon[:, c])
                        for c in range(raw.shape[1])], axis=0)
    d = raw.astype(np.float64) - recon.astype(np.float64)
    dIxx = d[:, 1:-1, 2:]  + d[:, 1:-1, :-2]  - 2 * d[:, 1:-1, 1:-1]
    dIyy = d[:, 2:,  1:-1] + d[:, :-2,  1:-1] - 2 * d[:, 1:-1, 1:-1]
    dIxy = (d[:, 2:, 2:] - d[:, 2:, :-2] - d[:, :-2, 2:] + d[:, :-2, :-2]) / 4
    return np.sqrt(dIxx**2 + 2*dIxy**2 + dIyy**2).mean(axis=(-2,-1)).astype(np.float32)


# ── inference for external datasets ──────────────────────────────────────────

def _infer(model, patch_dir: Path, device: str, batch_size: int = 256):
    ds = PatchDataset(str(patch_dir), condition=0, condition_name="")
    if len(ds) == 0:
        return np.array([]), np.array([])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, drop_last=False)
    cls_name = type(model).__name__
    raws, recons = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            if x.dim() == 3:
                x = x.unsqueeze(1)
            if "SemiSup" in cls_name:
                x_hat, _, _ = model(x)
            elif "Contrastive" in cls_name or "SupCon" in cls_name:
                x_hat, _ = model(x)
            else:
                x_hat, _ = model(x)
            for r, p in zip(x.cpu().numpy(), x_hat.cpu().numpy()):
                raws.append(r[0] if r.shape[0] == 1 else r)
                recons.append(p[0] if p.shape[0] == 1 else p)
    return np.stack(raws), np.stack(recons)


def _infer_2ch(model, ch1_dir: Path, ch3_dir: Path, device: str,
               batch_size: int = 256):
    """Run 2-channel model; return stacked (N, 2, H, W) raw and recon arrays."""
    ds = MultiChannelPatchDataset([str(ch1_dir), str(ch3_dir)],
                                  condition=0, condition_name="")
    if len(ds) == 0:
        return np.array([]), np.array([])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, drop_last=False)
    cls_name = type(model).__name__
    raws, recons = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            if "SemiSup" in cls_name:
                x_hat, _, _ = model(x)
            else:
                x_hat, _ = model(x)
            raws.append(x.cpu().numpy())
            recons.append(x_hat.cpu().numpy())
    return np.concatenate(raws), np.concatenate(recons)


# ── panel ─────────────────────────────────────────────────────────────────────

def _make_panel_2ch(raw_patches, recon_patches, patch_labels,
                    title, save_path, cols=COLS):
    """2-channel panel: 4 rows per slot — raw_pax / raw_act / recon_pax / recon_act."""
    n          = len(raw_patches)
    n_slots    = min(N_PATCHES, n)
    rows_slots = (n_slots + cols - 1) // cols
    fig_rows   = rows_slots * 4

    ch_labels = ["pax (raw)", "act (raw)", "pax (recon)", "act (recon)"]
    fig, axes = plt.subplots(fig_rows, cols,
                             figsize=(cols * 1.1, fig_rows * 1.1))
    axes = np.array(axes).reshape(fig_rows, cols)
    for ax in axes.flat:
        ax.axis("off")

    for idx in range(n_slots):
        slot = idx // cols
        col  = idx  % cols
        r = raw_patches[idx]    # (2, H, W)
        p = recon_patches[idx]  # (2, H, W)
        vmin_pax = float(min(r[0].min(), p[0].min()))
        vmax_pax = float(max(r[0].max(), p[0].max())) or 1e-6
        vmin_act = float(min(r[1].min(), p[1].min()))
        vmax_act = float(max(r[1].max(), p[1].max())) or 1e-6

        axes[slot*4,   col].imshow(r[0], cmap="gray", vmin=vmin_pax, vmax=vmax_pax)
        axes[slot*4,   col].set_title(patch_labels[idx], fontsize=5, pad=2)
        axes[slot*4+1, col].imshow(r[1], cmap="gray", vmin=vmin_act, vmax=vmax_act)
        axes[slot*4+2, col].imshow(p[0], cmap="gray", vmin=vmin_pax, vmax=vmax_pax)
        axes[slot*4+3, col].imshow(p[1], cmap="gray", vmin=vmin_act, vmax=vmax_act)

    for slot in range(rows_slots):
        for row_offset, lbl in enumerate(ch_labels):
            axes[slot*4 + row_offset, 0].text(
                -0.15, 0.5, lbl, fontsize=6, fontweight="bold",
                ha="right", va="center",
                transform=axes[slot*4 + row_offset, 0].transAxes)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _make_panel(raw_patches, recon_patches, patch_labels,
                title, save_path, cols=COLS):
    n          = len(raw_patches)
    rows_pairs = (N_PATCHES + cols - 1) // cols
    fig_rows   = rows_pairs * 2

    fig, axes = plt.subplots(fig_rows, cols,
                             figsize=(cols * 1.1, fig_rows * 1.1))
    axes = np.array(axes).reshape(fig_rows, cols)
    for ax in axes.flat:
        ax.axis("off")

    for idx in range(n):
        pair = idx // cols
        col  = idx  % cols
        r = raw_patches[idx];   p = recon_patches[idx]
        if r.ndim == 3 and r.shape[0] == 1:
            r = r[0]; p = p[0]
        vmin = float(min(r.min(), p.min()))
        vmax = float(max(r.max(), p.max()))
        if vmax <= vmin:
            vmax = vmin + 1e-6

        axes[pair*2,   col].imshow(r, cmap="gray", vmin=vmin, vmax=vmax)
        axes[pair*2,   col].set_title(patch_labels[idx], fontsize=5, pad=2)
        axes[pair*2+1, col].imshow(p, cmap="gray", vmin=vmin, vmax=vmax)

    for pair in range(rows_pairs):
        axes[pair*2,   0].text(-0.15, 0.5, "raw",   fontsize=7, fontweight="bold",
                                ha="right", va="center",
                                transform=axes[pair*2,   0].transAxes)
        axes[pair*2+1, 0].text(-0.15, 0.5, "recon", fontsize=7, fontweight="bold",
                                ha="right", va="center",
                                transform=axes[pair*2+1, 0].transAxes)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant_dir", type=Path)
    parser.add_argument("--subset",     default="unlabelled",
                        help="Label for output dir prefix (e.g. unlabelled, ppax)")
    parser.add_argument("--patch-dirs", nargs="+", type=Path, default=[],
                        help="External patch dirs for inference mode (ch1 for 1ch models)")
    parser.add_argument("--ch3-patch-dirs", nargs="+", type=Path, default=[],
                        help="ch3 patch dirs paired with --patch-dirs for 2ch models")
    parser.add_argument("--device",     default="auto")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    two_channel = bool(args.ch3_patch_dirs)

    vdir    = args.variant_dir
    out_dir = vdir / "quality_panels_bulk"
    out_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else args.device

    # ── get raw / recon arrays and per-patch labels ───────────────────────────
    if args.patch_dirs:
        # ── external inference mode ───────────────────────────────────────────
        model_pt = vdir / "model_best.pt"
        if not model_pt.exists():
            model_pt = vdir / "model_final.pt"
        if not model_pt.exists():
            sys.exit(f"No model checkpoint in {vdir}")

        print(f"Loading model from {model_pt.name} …", flush=True)
        model = torch.load(str(model_pt), map_location=device, weights_only=False)
        model.to(device).eval()

        all_raw, all_recon, all_labels = [], [], []
        ch3_dirs = list(args.ch3_patch_dirs) if two_channel else []
        for i, pd_path in enumerate(args.patch_dirs):
            cond = pd_path.parent.name   # e.g. "control" or "ycomp"
            print(f"  Inferring {pd_path} …", flush=True)
            if two_channel and i < len(ch3_dirs):
                r, p = _infer_2ch(model, pd_path, ch3_dirs[i], device)
            else:
                r, p = _infer(model, pd_path, device)
            if len(r) == 0:
                print(f"    [skip] no patches"); continue
            all_raw.append(r); all_recon.append(p)
            all_labels.extend([cond] * len(r))
            print(f"    {len(r)} patches", flush=True)

        if not all_raw:
            sys.exit("No patches found in provided --patch-dirs")

        raw_all   = np.concatenate(all_raw)
        recon_all = np.concatenate(all_recon)
        labels    = all_labels

    else:
        # ── vinc unlabelled mode ──────────────────────────────────────────────
        recon_dir = vdir / "recon"
        for p in [recon_dir / "patches_raw.tif",
                  recon_dir / "patches_recon.tif",
                  recon_dir / "patches_index.csv",
                  vdir / "latents.csv"]:
            if not p.exists():
                sys.exit(f"Required file not found: {p}")

        print("Loading TIFs …", flush=True)
        raw_full   = tifffile.imread(str(recon_dir / "patches_raw.tif"))
        recon_full = tifffile.imread(str(recon_dir / "patches_recon.tif"))

        idx_df = pd.read_csv(recon_dir / "patches_index.csv")
        lat_df = pd.read_csv(vdir / "latents.csv")
        lat_df["_name"] = lat_df["filename"].apply(lambda x: Path(x).stem)

        merged = idx_df.merge(
            lat_df[["_name", "annotation_label"]],
            left_on="name", right_on="_name", how="left"
        )
        unlabelled_mask = (merged["annotation_label"].isna() |
                           (merged["annotation_label"] == -1))
        sub = merged[unlabelled_mask].reset_index(drop=True)
        print(f"  Unlabelled patches: {len(sub)}", flush=True)

        frames    = sub["frame"].values
        raw_all   = raw_full[frames]
        recon_all = recon_full[frames]
        labels    = sub["split"].fillna("?").tolist()   # split comes from idx_df

    # detect 2ch from actual data shape (e.g. vinc unlabelled from 2ch recon TIFs)
    if raw_all.ndim == 4 and raw_all.shape[1] == 2:
        two_channel = True

    # ── compute metrics ───────────────────────────────────────────────────────
    print("Computing metrics …", flush=True)
    metric_arrays = {
        "recon_l1":         _compute_l1(raw_all, recon_all),
        "recon_mse":        _compute_mse(raw_all, recon_all),
        "recon_hessian_l1": _compute_hessian_l1(raw_all, recon_all),
    }

    # ── generate panels ───────────────────────────────────────────────────────
    for metric, vals in metric_arrays.items():
        mslug = _METRIC_SLUG.get(metric, metric)
        print(f"\n{metric}  min={vals.min():.4f}  "
              f"max={vals.max():.4f}  median={np.median(vals):.4f}", flush=True)

        for P in PERCENTILES:
            lo  = np.percentile(vals, max(P-1, 0))
            hi  = np.percentile(vals, min(P+1, 100))
            ctr = np.percentile(vals, P)

            in_win = np.where((vals >= lo) & (vals <= hi))[0]
            if len(in_win) == 0:
                print(f"  {P}p: no patches in window"); continue

            chosen  = rng.choice(in_win, size=min(N_PATCHES, len(in_win)),
                                 replace=False)
            chosen  = np.sort(chosen)
            plabels = [labels[i] for i in chosen]

            title = (f"{args.subset}  |  {mslug}  |  {P}th pct = {ctr:.4f}"
                     f"  [{lo:.4f}–{hi:.4f}]  n={len(chosen)}/{len(in_win)}")
            fname = out_dir / f"{args.subset}-{mslug}-{P}p.png"
            panel_fn = _make_panel_2ch if two_channel else _make_panel
            panel_fn(list(raw_all[chosen]), list(recon_all[chosen]),
                     plabels, title, fname)
            print(f"  {P}p  n={len(chosen):3d}/{len(in_win):4d}"
                  f"  center={ctr:.4f}  → {fname.name}", flush=True)

    print(f"\nDone. Panels in {out_dir}")


if __name__ == "__main__":
    main()
