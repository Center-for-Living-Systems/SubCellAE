#!/usr/bin/env python3
"""
make_recon_quality_panels_4ds.py

Reconstruction quality panels using GLOBAL percentile thresholds computed
from all four datasets (ds1=vinc, ds2=pfak, ds3=ppax, ds4=nih3t3) combined.

Each panel at percentile P shows ~30 patches sampled from ALL datasets whose
metric falls near the global Pth-percentile value.  Patches are labelled by
dataset/condition so the per-dataset distribution within each quality bucket
is visible.

Contrast with make_recon_quality_panels_bulk.py, where each subset computes
its own local percentile thresholds (so "10%" means different absolute values
across datasets).

Output: <variant_dir>/quality_panels_4ds_bulk/<metric>_<P>p.png
        <variant_dir>/quality_panels_4ds_bulk/global_percentiles.csv

Usage:
    python scripts/make_recon_quality_panels_4ds.py <variant_dir> \\
        [--root-folder /net/projects/CLS/lding/data/fa_data_analysis] \\
        [--device auto] [--batch-size 256] [--seed 42]
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
from subcellae.modelling.dataset import PatchDataset


N_PATCHES   = 30
COLS        = 6
PERCENTILES = list(range(10, 100, 10))

METRICS = ["recon_l1", "recon_mse", "recon_hessian_l1"]
_METRIC_SLUG = {
    "recon_l1":         "l1",
    "recon_mse":        "mse",
    "recon_hessian_l1": "hessian",
}

# External datasets relative to root_folder
EXTERNAL_DATASETS = [
    ("pfak",   "control", "ae_results/patches/cio_rb/pfak/control/tiff_patches32_mr10"),
    ("pfak",   "ycomp",   "ae_results/patches/cio_rb/pfak/ycomp/tiff_patches32_mr10"),
    ("ppax",   "control", "ae_results/patches/cio_rb/ppax/control/tiff_patches32_mr10"),
    ("ppax",   "ycomp",   "ae_results/patches/cio_rb/ppax/ycomp/tiff_patches32_mr10"),
    ("nih3t3", "control", "ae_results/patches/cio_rb/nih3t3/control/tiff_patches32_mr10"),
    ("nih3t3", "ycomp",   "ae_results/patches/cio_rb/nih3t3/ycomp/tiff_patches32_mr10"),
]

_DS_SHORT = {
    "vinc":   "ds1",
    "pfak":   "ds2",
    "ppax":   "ds3",
    "nih3t3": "ds4",
    "control": "ctrl",
    "ycomp":   "yc",
    "train":   "tr",
    "val":     "val",
}
def _short(s: str) -> str:
    return _DS_SHORT.get(s, s)


# ── metric helpers ─────────────────────────────────────────────────────────────

def _l1(raw, recon):
    return np.abs(raw.astype(np.float32) - recon.astype(np.float32)).mean(
        axis=tuple(range(1, raw.ndim)))

def _mse(raw, recon):
    d = raw.astype(np.float32) - recon.astype(np.float32)
    return (d * d).mean(axis=tuple(range(1, raw.ndim)))

def _hessian_l1(raw, recon):
    if raw.ndim == 4:
        return np.mean([_hessian_l1(raw[:, c], recon[:, c])
                        for c in range(raw.shape[1])], axis=0)
    d = raw.astype(np.float64) - recon.astype(np.float64)
    dIxx = d[:, 1:-1, 2:]  + d[:, 1:-1, :-2]  - 2 * d[:, 1:-1, 1:-1]
    dIyy = d[:, 2:,  1:-1] + d[:, :-2,  1:-1] - 2 * d[:, 1:-1, 1:-1]
    dIxy = (d[:, 2:, 2:] - d[:, 2:, :-2] - d[:, :-2, 2:] + d[:, :-2, :-2]) / 4
    return np.sqrt(dIxx**2 + 2*dIxy**2 + dIyy**2).mean(axis=(-2,-1)).astype(np.float32)

def _compute_metrics(raw, recon):
    return {
        "recon_l1":         _l1(raw, recon),
        "recon_mse":        _mse(raw, recon),
        "recon_hessian_l1": _hessian_l1(raw, recon),
    }


# ── inference ──────────────────────────────────────────────────────────────────

def _infer(model, patch_dir: Path, device: str, batch_size: int):
    ds = PatchDataset(str(patch_dir), condition=0, condition_name="")
    if len(ds) == 0:
        return np.array([]), np.array([])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, drop_last=False)
    cls_name = type(model).__name__
    raws, recons = [], []
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
    if not raws:
        return np.array([]), np.array([])
    return np.stack(raws), np.stack(recons)


# ── panel ──────────────────────────────────────────────────────────────────────

def _make_panel(raw_patches, recon_patches, patch_labels, title, save_path,
                cols=COLS):
    """6-col × N row-pair panel, raw on top, recon on bottom of each pair."""
    n = len(raw_patches)
    if n == 0:
        return
    n_slots    = min(N_PATCHES, n)
    rows_pairs = (n_slots + cols - 1) // cols
    fig_rows   = rows_pairs * 2

    fig, axes = plt.subplots(fig_rows, cols,
                             figsize=(cols * 1.1, fig_rows * 1.1))
    axes = np.array(axes).reshape(fig_rows, cols)
    for ax in axes.flat:
        ax.axis("off")

    for idx in range(n_slots):
        pair = idx // cols
        col  = idx  % cols
        r = raw_patches[idx]
        p = recon_patches[idx]
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
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant_dir", type=Path)
    parser.add_argument("--root-folder",
                        default="/net/projects/CLS/lding/data/fa_data_analysis",
                        type=Path)
    parser.add_argument("--device",     default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    vdir    = args.variant_dir
    out_dir = vdir / "quality_panels_4ds_bulk"
    out_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else args.device

    # ── 1. Load vinc from existing recon TIFs ─────────────────────────────────
    recon_dir = vdir / "recon"
    required  = [recon_dir / "patches_raw.tif",
                 recon_dir / "patches_recon.tif",
                 recon_dir / "patches_index.csv"]
    for p in required:
        if not p.exists():
            sys.exit(f"Required file not found: {p}")

    print("Loading vinc TIFs …", flush=True)
    raw_vinc   = tifffile.imread(str(recon_dir / "patches_raw.tif"))
    recon_vinc = tifffile.imread(str(recon_dir / "patches_recon.tif"))
    idx_df     = pd.read_csv(recon_dir / "patches_index.csv")

    # Build per-patch label for vinc: "ds1/{condition_name}/{split}"
    vinc_labels = []
    for _, row in idx_df.iterrows():
        cname = str(row.get("condition_name", "?"))
        split = str(row.get("split", "?"))
        # Shorten: keep dataset prefix + split, drop long condition name
        ds_part   = cname.split("_")[0] if "_" in cname else cname
        vinc_labels.append(f"ds1/{_short(ds_part)}/{_short(split)}")

    print(f"  vinc: {len(raw_vinc)} patches", flush=True)

    # ── 2. Load model for external inference ─────────────────────────────────
    model_pt = vdir / "model_final.pt"
    if not model_pt.exists():
        model_pt = vdir / "model_best.pt"
    if not model_pt.exists():
        sys.exit(f"No model checkpoint in {vdir}")

    print(f"Loading model {model_pt.name} …", flush=True)
    model = torch.load(str(model_pt), map_location=device, weights_only=False)
    model.to(device).eval()

    # ── 3. Run inference on pfak / ppax / nih3t3 ─────────────────────────────
    ext_raws:   list[np.ndarray] = []
    ext_recons: list[np.ndarray] = []
    ext_labels: list[str]        = []
    ext_groups: list[str]        = []  # ds_cond tag per patch (for windowing)

    for ds_name, cond_name, rel_path in EXTERNAL_DATASETS:
        patch_dir = args.root_folder / rel_path
        if not patch_dir.exists():
            print(f"  [skip] not found: {patch_dir}", flush=True)
            continue
        n_tifs = len(list(patch_dir.glob("*.tif")))
        print(f"  Inferring {ds_name}/{cond_name}  ({n_tifs} patches) …",
              flush=True)
        raw_e, recon_e = _infer(model, patch_dir, device, args.batch_size)
        if len(raw_e) == 0:
            print("    [skip] empty", flush=True); continue
        tag = f"{_DS_SHORT.get(ds_name, ds_name)}/{_short(cond_name)}"
        ext_raws.append(raw_e)
        ext_recons.append(recon_e)
        ext_labels.extend([tag] * len(raw_e))
        ext_groups.extend([tag] * len(raw_e))
        print(f"    {len(raw_e)} patches", flush=True)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # ── 4. Combine all data ───────────────────────────────────────────────────
    all_raw    = np.concatenate([raw_vinc] + ext_raws,   axis=0)
    all_recon  = np.concatenate([recon_vinc] + ext_recons, axis=0)
    all_labels = vinc_labels + ext_labels

    # dataset group tag per patch (for info only, not used in windowing)
    vinc_groups = []
    for row in idx_df.itertuples():
        cname = str(getattr(row, "condition_name", "?"))
        ds_part = cname.split("_")[0] if "_" in cname else cname
        split   = str(getattr(row, "split", "?"))
        vinc_groups.append(f"ds1/{_short(ds_part)}/{_short(split)}")
    all_groups = vinc_groups + ext_groups

    print(f"\nTotal patches: {len(all_raw)}", flush=True)
    n_vinc = len(raw_vinc)
    n_ext  = len(all_raw) - n_vinc
    print(f"  vinc: {n_vinc}  |  external: {n_ext}", flush=True)

    # ── 5. Compute metrics for all patches ────────────────────────────────────
    print("Computing metrics …", flush=True)
    metrics = _compute_metrics(all_raw, all_recon)

    # ── 6. Global percentile thresholds ───────────────────────────────────────
    pct_rows = []
    for metric, vals in metrics.items():
        mslug = _METRIC_SLUG.get(metric, metric)
        print(f"\n{metric}: global min={vals.min():.4f}  "
              f"median={np.median(vals):.4f}  max={vals.max():.4f}", flush=True)

        # per-percentile window: use the global ±1 percentile points
        for P in PERCENTILES:
            global_lo  = np.percentile(vals, max(P - 1, 0))
            global_hi  = np.percentile(vals, min(P + 1, 100))
            global_ctr = np.percentile(vals, P)
            pct_rows.append({"metric": metric, "percentile": P,
                              "value": global_ctr,
                              "lo": global_lo, "hi": global_hi})

            in_win = np.where((vals >= global_lo) & (vals <= global_hi))[0]
            if len(in_win) == 0:
                print(f"  {P}p: no patches in global window — skipping",
                      flush=True)
                continue

            # sample up to N_PATCHES from the window
            n_draw = min(N_PATCHES, len(in_win))
            chosen = rng.choice(in_win, size=n_draw, replace=False)
            chosen = chosen[np.argsort(vals[chosen])]   # sort by metric value

            # break down by group for info
            groups_in = [all_groups[i] for i in chosen]
            from collections import Counter
            cnt = Counter(groups_in)
            cnt_str = "  ".join(f"{g}:{c}" for g, c in sorted(cnt.items()))

            title = (f"{mslug}  |  global {P}th pct = {global_ctr:.4f}"
                     f"  [{global_lo:.4f}–{global_hi:.4f}]"
                     f"  n={len(chosen)}/{len(in_win)}\n{cnt_str}")

            fname = out_dir / f"{mslug}_{P:02d}p.png"
            _make_panel(
                list(all_raw[chosen]), list(all_recon[chosen]),
                [all_labels[i] for i in chosen], title, fname
            )
            print(f"  {P}p  n={len(chosen):3d}/{len(in_win):5d}"
                  f"  ctr={global_ctr:.4f}  → {fname.name}", flush=True)

    # ── 7. Save global percentile table ──────────────────────────────────────
    pd.DataFrame(pct_rows).to_csv(out_dir / "global_percentiles.csv", index=False)
    print(f"\nGlobal percentile table → {out_dir / 'global_percentiles.csv'}")
    print(f"Done. Panels in {out_dir}")


if __name__ == "__main__":
    main()
