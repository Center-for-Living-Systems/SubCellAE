#!/usr/bin/env python3
"""
Violin plots of reconstruction metrics (L1, MSE, Hessian L1) with decile lines.

Produces two sets of plots per model:
  1. vinc — FA type × train/val split (labelled patches only, No adhesion excluded)
  2. ppax — condition (control / ycomp) using model inference

Each violin has 9 horizontal lines at the 10th–90th percentiles.

Output: <variant_dir>/violin_plots/
  vinc_{metric}.png
  ppax_{metric}.png

Usage
-----
  # vinc only (uses existing latents.csv)
  python scripts/plot_recon_metric_violins.py <variant_dir>

  # vinc + ppax
  python scripts/plot_recon_metric_violins.py <variant_dir> \\
      --patch-dirs \\
        /path/to/ppax/control/tiff_patches32 \\
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
import matplotlib.patches as mpatches
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parents[1]))
from subcellae.modelling.dataset import PatchDataset


METRICS = {
    "recon_l1":         "Reconstruction L1 (MAE)",
    "recon_mse":        "Reconstruction MSE",
    "recon_hessian_l1": "Hessian L1",
}
FA_ORDER   = ["Nascent Adhesion", "focal complex", "focal adhesion", "fibrillar adhesion"]
FA_SHORT   = {"Nascent Adhesion": "Nascent", "focal complex": "Focal\nComplex",
              "focal adhesion":   "Focal\nAdhesion", "fibrillar adhesion": "Fibrillar"}
SPLIT_COLOR = {"train": "#4C72B0", "val": "#DD8452"}
COND_COLOR  = {"control": "#55A868", "ycomp": "#C44E52"}

VIOLIN_WIDTH = 0.35
GROUP_GAP    = 1.0    # gap between FA groups
WITHIN_GAP   = 0.4   # gap between train/val within a group
PCT_LINEWIDTH = 0.9
PCT_ALPHA     = 0.65


# ── helpers ───────────────────────────────────────────────────────────────────

def _compute_metrics(raw: np.ndarray, recon: np.ndarray) -> dict:
    d = raw.astype(np.float32) - recon.astype(np.float32)
    l1   = np.abs(d).mean(axis=tuple(range(1, raw.ndim)))
    mse  = (d * d).mean(axis=tuple(range(1, raw.ndim)))
    if raw.ndim == 3:   # (N, H, W)
        dd = raw.astype(np.float64) - recon.astype(np.float64)
        dxx = dd[:, 1:-1, 2:]  + dd[:, 1:-1, :-2]  - 2*dd[:, 1:-1, 1:-1]
        dyy = dd[:, 2:,  1:-1] + dd[:, :-2,  1:-1] - 2*dd[:, 1:-1, 1:-1]
        dxy = (dd[:, 2:,2:] - dd[:, 2:,:-2] - dd[:, :-2,2:] + dd[:, :-2,:-2])/4
        hess = np.sqrt(dxx**2 + 2*dxy**2 + dyy**2).mean(axis=(-2,-1)).astype(np.float32)
    else:
        hess = np.zeros(len(raw), dtype=np.float32)
    return {"recon_l1": l1, "recon_mse": mse, "recon_hessian_l1": hess}


def _infer(model, patch_dir: Path, device: str, batch_size=256):
    ds = PatchDataset(str(patch_dir), condition=0, condition_name="")
    if len(ds) == 0:
        return np.array([]), np.array([])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, drop_last=False)
    raws, recons = [], []
    cls_name = type(model).__name__
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            if x.dim() == 3:
                x = x.unsqueeze(1)
            if "SemiSup" in cls_name:
                x_hat, _, _ = model(x)
            else:
                x_hat, _ = model(x)
            for r, p in zip(x.cpu().numpy(), x_hat.cpu().numpy()):
                raws.append(r[0] if r.shape[0] == 1 else r)
                recons.append(p[0] if p.shape[0] == 1 else p)
    return np.stack(raws), np.stack(recons)


def _draw_violin_with_pcts(ax, data: np.ndarray, pos: float,
                            color: str, ref_pcts: np.ndarray,
                            width: float = VIOLIN_WIDTH,
                            label: str = None) -> None:
    """Draw one violin + 9 decile lines at position `pos`.

    ref_pcts : pre-computed percentiles from the full FA/condition pool so that
               train and val violins for the same group share identical lines.
    """
    if len(data) < 2:
        return
    parts = ax.violinplot([data], positions=[pos], widths=[width],
                           showmeans=False, showmedians=False, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_linewidth(0.5)
        body.set_alpha(0.7)
        if label:
            body.set_label(label)

    ax.hlines(ref_pcts,
              pos - width * 0.45,
              pos + width * 0.45,
              colors="black", linewidths=PCT_LINEWIDTH, alpha=PCT_ALPHA, zorder=3)


# ── vinc plot ─────────────────────────────────────────────────────────────────

def plot_vinc(df: pd.DataFrame, out_dir: Path) -> None:
    """One figure per metric: FA type × train/val violins."""
    for metric, ylabel in METRICS.items():
        if metric not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(11, 5))

        tick_positions, tick_labels = [], []
        legend_handles = []
        seen_splits = set()

        for gi, fa in enumerate(FA_ORDER):
            fa_df = df[df["annotation_label_name"] == fa]
            if fa_df.empty:
                continue

            # Percentiles from the full FA pool (train + val combined)
            pool_vals = fa_df[metric].dropna().values
            ref_pcts  = np.percentile(pool_vals, range(10, 100, 10))

            group_center = gi * (2 * VIOLIN_WIDTH + WITHIN_GAP + GROUP_GAP)
            splits_present = [s for s in ["train", "val"] if s in fa_df["split"].values]

            for si, split in enumerate(splits_present):
                pos = group_center + si * (VIOLIN_WIDTH + WITHIN_GAP * 0.5)
                data = fa_df[fa_df["split"] == split][metric].dropna().values
                lbl  = split if split not in seen_splits else None
                _draw_violin_with_pcts(ax, data, pos, SPLIT_COLOR[split],
                                       ref_pcts=ref_pcts, label=lbl)
                if split not in seen_splits:
                    legend_handles.append(
                        mpatches.Patch(color=SPLIT_COLOR[split], label=split, alpha=0.7))
                    seen_splits.add(split)

            # group tick at center of the two violins
            n_splits = len(splits_present)
            center = group_center + (n_splits - 1) * (VIOLIN_WIDTH + WITHIN_GAP * 0.5) / 2
            tick_positions.append(center)
            tick_labels.append(FA_SHORT.get(fa, fa))

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"vinc — {ylabel}  (labelled, No adhesion excluded)\n"
                     f"lines = 10th–90th percentile", fontsize=11)
        ax.legend(handles=legend_handles, title="split", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()

        out = out_dir / f"vinc_{metric}.png"
        fig.savefig(str(out), dpi=150)
        plt.close(fig)
        print(f"  saved {out.name}")


# ── ppax plot ─────────────────────────────────────────────────────────────────

def plot_ppax(cond_metrics: dict, out_dir: Path) -> None:
    """cond_metrics: {condition_name: {metric: array}}. One figure per metric."""
    conds = list(cond_metrics.keys())

    for metric, ylabel in METRICS.items():
        fig, ax = plt.subplots(figsize=(5, 5))

        tick_positions, tick_labels = [], []
        legend_handles = []

        # Percentiles from the full ppax pool (control + ycomp combined)
        all_ppax = np.concatenate([v[metric] for v in cond_metrics.values()
                                   if metric in v and len(v[metric]) > 0])
        ref_pcts = np.percentile(all_ppax, range(10, 100, 10))

        for ci, cond in enumerate(conds):
            data = cond_metrics[cond].get(metric)
            if data is None or len(data) < 2:
                continue
            pos   = float(ci)
            color = COND_COLOR.get(cond, "#999999")
            _draw_violin_with_pcts(ax, data, pos, color, ref_pcts=ref_pcts)
            legend_handles.append(
                mpatches.Patch(color=color, label=cond, alpha=0.7))
            tick_positions.append(pos)
            tick_labels.append(cond)

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"ppax — {ylabel}\nlines = 10th–90th percentile", fontsize=11)
        ax.legend(handles=legend_handles, fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()

        out = out_dir / f"ppax_{metric}.png"
        fig.savefig(str(out), dpi=150)
        plt.close(fig)
        print(f"  saved {out.name}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant_dir", type=Path)
    parser.add_argument("--patch-dirs", nargs="+", type=Path, default=[],
                        help="ppax patch dirs for inference (optional)")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    vdir    = args.variant_dir
    out_dir = vdir / "violin_plots"
    out_dir.mkdir(exist_ok=True)

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else args.device

    # ── vinc ─────────────────────────────────────────────────────────────────
    lat_csv = vdir / "latents.csv"
    if not lat_csv.exists():
        sys.exit(f"latents.csv not found in {vdir}")

    lat = pd.read_csv(lat_csv)
    df  = lat[(lat["annotation_label"] != -1) &
              (lat["annotation_label_name"] != "No adhesion")].copy()
    print(f"vinc labelled patches (excl. No adhesion): {len(df)}")
    print(df.groupby(["annotation_label_name", "split"]).size().to_string())
    print()
    plot_vinc(df, out_dir)

    # ── ppax ─────────────────────────────────────────────────────────────────
    if args.patch_dirs:
        model_pt = vdir / "model_best.pt"
        if not model_pt.exists():
            model_pt = vdir / "model_final.pt"
        print(f"\nLoading model from {model_pt.name} …")
        model = torch.load(str(model_pt), map_location=device, weights_only=False)
        model.to(device).eval()

        cond_metrics = {}
        for pd_path in args.patch_dirs:
            cond = pd_path.parent.name
            print(f"  Inferring {cond} ({pd_path.name}) …", flush=True)
            raw, recon = _infer(model, pd_path, device)
            if len(raw) == 0:
                print(f"    [skip] no patches"); continue
            print(f"    {len(raw)} patches", flush=True)
            cond_metrics[cond] = _compute_metrics(raw, recon)

        if cond_metrics:
            plot_ppax(cond_metrics, out_dir)

    print(f"\nDone. Plots in {out_dir}")


if __name__ == "__main__":
    main()
