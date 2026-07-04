#!/usr/bin/env python3
"""
Visualise reconstruction quality by metric × percentile × FA type.

For each combination of metric / decile / FA type, creates one panel figure
showing up to 20 labelled patches (No adhesion excluded):

    4 pairs of rows × 5 columns  (=20 slots)
    top row of each pair    = recon
    bottom row of each pair = raw
    Empty slots get axis-off blanks.

Percentile windows: P ∈ {10,20,...,90},  window = [pct(P-1), pct(P+1)]
computed over the FA-type subset.

Output filenames:
  <variant_dir>/quality_panels/<metric>_pct<P>_<fa_type_slug>.png

Reads
-----
  <variant_dir>/recon/patches_raw.tif
  <variant_dir>/recon/patches_recon.tif
  <variant_dir>/recon/patches_index.csv   (frame → name)
  <variant_dir>/latents.csv               (name → metrics + annotation labels)

Usage
-----
  python scripts/make_recon_quality_panels.py <variant_dir>
  python scripts/make_recon_quality_panels.py <variant_dir> --cols 5 --seed 42
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


EXCLUDE_LABELS = {"No adhesion"}
N_PATCHES      = 6
COLS           = 6
PERCENTILES    = list(range(10, 100, 10))   # 10,20,...,90
METRICS        = ["recon_l1", "recon_mse", "recon_hessian_l1"]

# Short names used in output filenames
_FA_SLUG = {
    "Nascent Adhesion":   "nascent",
    "focal complex":      "focalcomplex",
    "focal adhesion":     "focalad",
    "fibrillar adhesion": "fibrillar",
}
_METRIC_SLUG = {
    "recon_l1":          "l1",
    "recon_mse":         "mse",
    "recon_hessian_l1":  "hessian",
}

def _slug(name: str) -> str:
    return _FA_SLUG.get(name, name.lower().replace(" ", "").replace("/", ""))


def _make_panel(raw_patches: list, recon_patches: list, splits: list,
                title: str, save_path: Path,
                cols: int = COLS) -> None:
    """
    Two rows × cols: row 0 = recon, row 1 = raw.
    Each column gets a small 'train'/'val' title above the recon row.
    Empty slots remain blank (axis off).
    """
    n = len(raw_patches)

    fig, axes = plt.subplots(2, cols, figsize=(cols * 1.1, 2 * 1.1))
    axes = np.array(axes).reshape(2, cols)

    for ax in axes.flat:
        ax.axis("off")

    for idx in range(n):
        raw_p = raw_patches[idx]
        rec_p = recon_patches[idx]
        if raw_p.ndim == 3 and raw_p.shape[0] == 1:
            raw_p = raw_p[0]; rec_p = rec_p[0]

        vmin = float(min(raw_p.min(), rec_p.min()))
        vmax = float(max(raw_p.max(), rec_p.max()))
        if vmax <= vmin:
            vmax = vmin + 1e-6

        axes[0, idx].imshow(raw_p,  cmap="gray", vmin=vmin, vmax=vmax)
        axes[0, idx].set_title(splits[idx], fontsize=6, pad=2)
        axes[1, idx].imshow(rec_p, cmap="gray", vmin=vmin, vmax=vmax)

    # Row labels on the left — use text() so they survive axis("off")
    for row, label in [(0, "raw"), (1, "recon")]:
        axes[row, 0].text(-0.15, 0.5, label, fontsize=8, fontweight="bold",
                          ha="right", va="center",
                          transform=axes[row, 0].transAxes)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant_dir", type=Path)
    parser.add_argument("--cols",  type=int, default=COLS)
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()

    vdir      = args.variant_dir
    recon_dir = vdir / "recon"

    for p in [recon_dir / "patches_raw.tif",
              recon_dir / "patches_recon.tif",
              recon_dir / "patches_index.csv",
              vdir / "latents.csv"]:
        if not p.exists():
            sys.exit(f"Required file not found: {p}")

    out_dir = vdir / "quality_panels"
    out_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # ── load data ─────────────────────────────────────────────────────────────
    print("Loading TIFs …", flush=True)
    raw_all   = tifffile.imread(str(recon_dir / "patches_raw.tif"))
    recon_all = tifffile.imread(str(recon_dir / "patches_recon.tif"))
    print(f"  {raw_all.shape[0]} patches, shape {raw_all.shape}", flush=True)

    idx_df = pd.read_csv(recon_dir / "patches_index.csv")   # frame, name, ...
    lat_df = pd.read_csv(vdir / "latents.csv")
    lat_df["_name"] = lat_df["filename"].apply(lambda p: Path(p).stem)

    # join patches_index → latents on name
    merged = idx_df.merge(
        lat_df[["_name", "annotation_label", "annotation_label_name"] + METRICS],
        left_on="name", right_on="_name", how="left"
    )

    # ── filter to labelled, non-excluded ─────────────────────────────────────
    labelled = (
        merged["annotation_label"].notna() &
        (merged["annotation_label"] != -1) &
        (~merged["annotation_label_name"].isin(EXCLUDE_LABELS))
    )
    df = merged[labelled].copy().reset_index(drop=True)
    print(f"  Labelled (excl. No adhesion): {len(df)} patches", flush=True)
    print(df["annotation_label_name"].value_counts().to_string(), flush=True)

    fa_types = sorted(df["annotation_label_name"].dropna().unique())

    # ── generate panels ───────────────────────────────────────────────────────
    for metric in METRICS:
        if metric not in df.columns:
            print(f"  [skip] {metric} not in data"); continue

        print(f"\n{metric}", flush=True)

        # Global percentile boundaries from ALL labelled vinc patches pooled
        # (across all FA types, train+val, control+ycomp)
        global_vals = df[metric].dropna().values.astype(np.float32)
        # ±2.5 percentile-point window — wide enough to catch small FA classes
        global_pct  = {P: (np.percentile(global_vals, max(P-2.5, 0)),
                           np.percentile(global_vals, P),
                           np.percentile(global_vals, min(P+2.5, 100)))
                       for P in PERCENTILES}
        print(f"  global range: [{global_vals.min():.4f}, {global_vals.max():.4f}]  "
              f"median={np.median(global_vals):.4f}", flush=True)

        for fa in fa_types:
            fa_df = df[df["annotation_label_name"] == fa].copy()

            if len(fa_df) < 2:
                print(f"  {fa}: too few patches ({len(fa_df)}), skip"); continue

            slug = _slug(fa)
            print(f"  {fa} (n={len(fa_df)})", flush=True)

            for P in PERCENTILES:
                lo_val, center, hi_val = global_pct[P]

                in_win = fa_df[(fa_df[metric] >= lo_val) &
                               (fa_df[metric] <= hi_val)].copy()

                if len(in_win) == 0:
                    print(f"    pct{P:02d}: 0 patches in window, skip")
                    continue

                # take up to N_PATCHES; use all if fewer available
                n_take = min(N_PATCHES, len(in_win))
                chosen = in_win.sample(n=n_take, random_state=args.seed)

                frames = chosen["frame"].values
                raw_p   = [raw_all[f]   for f in frames]
                recon_p = [recon_all[f] for f in frames]
                splits  = chosen["split"].fillna("?").tolist()

                metric_short = _METRIC_SLUG.get(metric, metric)
                title = (f"{fa}  |  {metric_short}  |  {P}th pct = {center:.4f}"
                         f"  [{lo_val:.4f}–{hi_val:.4f}]  n={n_take}/{len(in_win)}")
                fname = out_dir / f"{slug}-{metric_short}-{P}p.png"
                _make_panel(raw_p, recon_p, splits, title, fname, cols=args.cols)
                print(f"    {P}p  n={n_take}/{len(in_win)}"
                      f"  center={center:.4f}  → {fname.name}", flush=True)

    print(f"\nDone. Panels in {out_dir}")


if __name__ == "__main__":
    main()
