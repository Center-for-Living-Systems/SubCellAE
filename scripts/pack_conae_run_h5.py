#!/usr/bin/env python3
"""
pack_conae_run_h5.py
====================
Pack one ConAE / SupConAE model result directory into model.h5.

H5 layout
---------
  meta/latents_csv             bytes  — latents.csv
  meta/cluster_labels_csv      bytes  — eval/cluster_panels/cluster_labels.csv
  meta/eval_summary_csv        bytes  — eval/eval_summary.csv (if present)
  meta/cross_ds_metrics_csv    bytes  — cross_dataset_recon_metrics.csv (if present)
  meta/cross_ds_latents_csv    bytes  — eval/cross_dataset_latents.csv (if present)
  cluster_panels/{name}        float32 — each cluster panel TIF in eval/cluster_panels/
  cluster_panels_proj/{name}   float32 — each cluster panel TIF in eval/cluster_panels_proj/
  plots/{name}                 uint8 bytes — every PNG under eval/
  plots_toplevel/{name}        uint8 bytes — every PNG at result_dir top level

attrs: model_name, result_dir, n_patches

Usage
-----
  python scripts/pack_conae_run_h5.py <result_dir>
  python scripts/pack_conae_run_h5.py <result_dir> --out /tmp/model.h5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import tifffile  # still needed for cluster panel TIFs


def _pack_csv(hf: h5py.File, key: str, path: Path) -> bool:
    if not path.exists():
        return False
    hf.create_dataset(key, data=np.bytes_(path.read_text()))
    return True


def _pack_tif(hf: h5py.File, key: str, path: Path) -> bool:
    if not path.exists():
        return False
    arr = tifffile.imread(str(path)).astype(np.float32)
    hf.create_dataset(key, data=arr, compression="gzip", compression_opts=4)
    return True


def _pack_png(hf: h5py.File, key: str, path: Path) -> None:
    hf.create_dataset(key, data=np.frombuffer(path.read_bytes(), dtype=np.uint8))


def pack(result_dir: Path, out_h5: Path) -> None:
    if not result_dir.is_dir():
        sys.exit(f"ERROR: {result_dir} is not a directory")

    print(f"[pack] {result_dir.name}")
    print(f"[pack] → {out_h5}")

    eval_dir  = result_dir / "eval"
    recon_dir = result_dir / "recon"

    with h5py.File(str(out_h5), "w") as hf:

        # ── meta CSVs ────────────────────────────────────────────────────────
        lat_csv = result_dir / "latents.csv"
        if not lat_csv.exists():
            sys.exit(f"ERROR: latents.csv not found in {result_dir}")
        import pandas as pd
        df = pd.read_csv(lat_csv)
        hf.create_dataset("meta/latents_csv", data=np.bytes_(lat_csv.read_text()))
        print(f"[pack]   latents.csv  ({len(df)} patches)")

        ok = _pack_csv(hf, "meta/cluster_labels_csv",
                       eval_dir / "cluster_panels" / "cluster_labels.csv")
        print(f"[pack]   cluster_labels.csv  {'OK' if ok else 'MISSING'}")

        _pack_csv(hf, "meta/eval_summary_csv",        eval_dir / "eval_summary.csv")
        _pack_csv(hf, "meta/cross_ds_metrics_csv",
                  result_dir / "cross_dataset_recon_metrics.csv")
        ok = _pack_csv(hf, "meta/cross_ds_latents_csv",
                       eval_dir / "cross_dataset_latents.csv")
        if ok:
            print(f"[pack]   cross_dataset_latents.csv  OK")

        # ── cluster panels (z_recon) ──────────────────────────────────────────
        cp_dir = eval_dir / "cluster_panels"
        n_panels = 0
        if cp_dir.is_dir():
            for tif in sorted(cp_dir.glob("*.tif")):
                stem = tif.stem
                _pack_tif(hf, f"cluster_panels/{stem}", tif)
                n_panels += 1
        print(f"[pack]   cluster_panels  {n_panels} TIFs")

        # ── cluster panels (z_proj) ───────────────────────────────────────────
        cp_proj_dir = eval_dir / "cluster_panels_proj"
        n_proj = 0
        if cp_proj_dir.is_dir():
            for tif in sorted(cp_proj_dir.glob("*.tif")):
                stem = tif.stem
                _pack_tif(hf, f"cluster_panels_proj/{stem}", tif)
                n_proj += 1
        if n_proj:
            print(f"[pack]   cluster_panels_proj  {n_proj} TIFs")

        # ── scatter plots (all PNGs under eval/) ──────────────────────────────
        n_plots = 0
        if eval_dir.is_dir():
            for png in sorted(eval_dir.glob("*.png")):
                _pack_png(hf, f"plots/{png.stem}", png)
                n_plots += 1
        print(f"[pack]   plots  {n_plots} PNGs")

        # ── top-level PNGs (loss curves, recon snapshots, cross_ds violin) ───
        n_toplevel = 0
        for png in sorted(result_dir.glob("*.png")):
            if png.name.startswith("._"):
                continue
            _pack_png(hf, f"plots_toplevel/{png.stem}", png)
            n_toplevel += 1
        print(f"[pack]   plots_toplevel  {n_toplevel} PNGs")

        # ── attrs ─────────────────────────────────────────────────────────────
        hf.attrs["model_name"] = result_dir.name
        hf.attrs["result_dir"] = str(result_dir)
        hf.attrs["n_patches"]  = int(len(df))

    size_mb = out_h5.stat().st_size / 1e6
    print(f"[pack]   → {size_mb:.1f} MB")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="Output H5 path (default: <result_dir>/model.h5)")
    args = ap.parse_args()

    result_dir = args.result_dir.resolve()
    out_h5     = args.out.resolve() if args.out else result_dir / "model.h5"
    pack(result_dir, out_h5)


if __name__ == "__main__":
    main()
