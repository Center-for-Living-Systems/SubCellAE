#!/usr/bin/env python3
"""
plot_cio_diff_histograms.py
============================
Histograms of (CIO - CIO_RB) pixel differences per dataset, PAX channel.
Shows the rolling-ball contribution: positive = pixels lowered by RB.
"""

from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

ROOT     = Path("/net/projects/CLS/lding/data/fa_data_analysis")
BASE_RB  = ROOT / "ae_results/patches/cio_rb"
BASE_CIO = ROOT / "ae_results/patches/cio"
PAX_CH   = 1

DATASETS = [
    {"key": "vinc",   "label": "ds1 - vinc",   "color": "#2E86C1"},
    {"key": "pfak",   "label": "ds2 - pfak",   "color": "#E74C3C"},
    {"key": "ppax",   "label": "ds3 - ppax",   "color": "#27AE60"},
    {"key": "nih3t3", "label": "ds4 - nih3t3", "color": "#8E44AD"},
]
CONDITIONS = ["control", "ycomp"]
PATCH_DIR  = "tiff_patches32_mr10"
N_BINS     = 200


def load_diffs(ds_key: str) -> np.ndarray:
    """Load matched CIO and CIO-RB patches, return (CIO - CIO_RB) pixel array."""
    diffs = []
    for cond in CONDITIONS:
        dir_cio = BASE_CIO / ds_key / cond / PATCH_DIR
        dir_rb  = BASE_RB  / ds_key / cond / PATCH_DIR
        if not dir_cio.exists() or not dir_rb.exists():
            print(f"  [skip] {ds_key}/{cond} — dir missing")
            continue
        names_cio = {p.name for p in dir_cio.glob("*.tif")}
        names_rb  = {p.name for p in dir_rb.glob("*.tif")}
        matched   = sorted(names_cio & names_rb)
        print(f"  {ds_key}/{cond}: {len(matched)} matched patches")
        for name in matched:
            arr_cio = tifffile.imread(dir_cio / name).astype(np.float32)
            arr_rb  = tifffile.imread(dir_rb  / name).astype(np.float32)
            # patches are (C, H, W) or (H, W); extract PAX channel if multi-channel
            if arr_cio.ndim == 3:
                arr_cio = arr_cio[PAX_CH]
                arr_rb  = arr_rb[PAX_CH]
            # patches already single-channel (32,32) from patchprep major_ch=1
            diffs.append((arr_cio - arr_rb).ravel())
    return np.concatenate(diffs) if diffs else np.array([])


def main():
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle("CIO − CIO_RB pixel difference per dataset (PAX channel)", fontsize=14)

    all_diffs = {}
    for ds in DATASETS:
        print(f"\n=== {ds['label']} ===")
        d = load_diffs(ds["key"])
        all_diffs[ds["key"]] = d
        if d.size == 0:
            continue

        mean_d = d.mean()
        std_d  = d.std()
        pct_nonzero = 100 * np.mean(np.abs(d) > 0.01)
        print(f"  mean={mean_d:.5f}  std={std_d:.5f}  |diff|>0.01: {pct_nonzero:.1f}%")

    # Row 0: full range histogram
    # Row 1: zoomed to [-0.05, 0.05]
    for col, ds in enumerate(DATASETS):
        d = all_diffs.get(ds["key"], np.array([]))
        color = ds["color"]
        label = ds["label"]

        for row, (ax, xlim) in enumerate(zip(axes[:, col], [None, (-0.05, 0.05)])):
            if d.size == 0:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                continue

            bins = np.linspace(d.min(), d.max(), N_BINS + 1) if xlim is None else \
                   np.linspace(xlim[0], xlim[1], N_BINS + 1)

            ax.hist(d, bins=bins, color=color, alpha=0.75, density=True, histtype="stepfilled")
            ax.axvline(0, color="black", lw=0.8, ls="--")
            ax.axvline(d.mean(), color="red", lw=1.2, ls="-", label=f"mean={d.mean():.4f}")

            if xlim:
                ax.set_xlim(xlim)

            pct = 100 * np.mean(np.abs(d) > 0.01)
            title = label if row == 0 else ""
            subtitle = f"std={d.std():.4f}  |Δ|>0.01: {pct:.1f}%"
            ax.set_title(f"{title}\n{subtitle}" if title else subtitle, fontsize=9)
            ax.set_xlabel("CIO − CIO_RB" if row == 1 else "")
            if xlim:
                ax.set_xlabel("CIO − CIO_RB  [zoomed]")
            ax.legend(fontsize=7)

    axes[0, 0].set_ylabel("density (full range)")
    axes[1, 0].set_ylabel("density (zoomed ±0.05)")

    plt.tight_layout()
    out = Path("cio_diff_histograms.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")

    # Also print a summary table
    print("\nDataset       n_pixels     mean       std     |Δ|>0.01(%)")
    print("-" * 65)
    for ds in DATASETS:
        d = all_diffs.get(ds["key"], np.array([]))
        if d.size:
            pct = 100 * np.mean(np.abs(d) > 0.01)
            print(f"{ds['label']:<14}  {d.size:>10,}  {d.mean():>9.5f}  {d.std():>8.5f}  {pct:>10.1f}%")


if __name__ == "__main__":
    main()
