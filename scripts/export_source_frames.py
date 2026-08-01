#!/usr/bin/env python3
"""
export_source_frames.py
=======================
Extract canvas frames from data.h5 and save them as per-channel TIF files
in the same layout as frameextract_pipeline output, so the existing
EnlargedCropDataset can load them with frame_dir.

Output layout:
  {output_root}/{norm}/{ds}/{cond}/{cond}_f{frame_idx:04d}_{ch}.tif

Usage:
    python scripts/export_source_frames.py                        # all datasets, cio_mode_prt
    python scripts/export_source_frames.py --norm cio_inlier      # different norm key
    python scripts/export_source_frames.py --datasets vinc pfak
    python scripts/export_source_frames.py --h5-root /path/to/patches/cio_inlier_med
"""
from __future__ import annotations

import argparse
import json
from io import StringIO
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import tifffile

DATA_ROOT  = Path("/net/projects/CLS/lding/data/fa_data_analysis")
H5_ROOT    = DATA_ROOT / "ae_results/patches/cio_inlier_med"
FRAME_ROOT = DATA_ROOT / "ae_results/source_frames"

ALL_DATASETS   = ["vinc", "nih3t3", "ppax", "pfak"]
ALL_NORM_KEYS  = ["cio_inlier", "cio_med", "cio_mode", "cio_mode_prt"]


def export_dataset(ds: str, norm_keys: list[str],
                   h5_root: Path, output_root: Path) -> None:
    h5_path = h5_root / ds / "data.h5"
    if not h5_path.exists():
        print(f"  SKIP {ds}: {h5_path} not found", flush=True)
        return

    with h5py.File(str(h5_path), "r") as f:
        # Load frame metadata
        img_meta = pd.read_csv(StringIO(f["images/meta"][()].decode()))
        channels  = json.loads(f.attrs["channels"])

        for nk in norm_keys:
            if f"images/{nk}" not in f:
                print(f"  SKIP {ds}/{nk}: group not in H5", flush=True)
                continue

            for ch in channels:
                ds_key = f"images/{nk}/{ch}"
                if ds_key not in f:
                    continue
                frames = f[ds_key][()]  # (M, H, W) float32

                for _, row in img_meta.iterrows():
                    frame_row = int(row["frame"])
                    cond      = str(row["condition_name"])
                    fi        = int(row["frame_idx"])

                    if frame_row >= len(frames):
                        print(f"    WARNING: frame_row={frame_row} out of range for {ds}/{nk}/{ch}")
                        continue

                    out_dir = output_root / nk / ds / cond
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"{cond}_f{fi:04d}_{ch}.tif"

                    tifffile.imwrite(
                        str(out_path),
                        frames[frame_row].astype(np.float32),
                        imagej=True,
                        metadata={"axes": "YX"},
                    )

            n_frames = len(img_meta)
            n_ch     = len(channels)
            print(f"  {ds}/{nk}: {n_frames} frames × {n_ch} channels exported", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets",  nargs="+", default=ALL_DATASETS, metavar="DS")
    ap.add_argument("--norm",      nargs="+", default=["cio_mode_prt"],
                    choices=ALL_NORM_KEYS, metavar="NORM",
                    help="Norm key(s) to export (default: cio_mode_prt)")
    ap.add_argument("--h5-root",   default=None, metavar="PATH",
                    help=f"Root containing {{ds}}/data.h5 (default: {H5_ROOT})")
    ap.add_argument("--output-root", default=None, metavar="PATH",
                    help=f"Output root for TIF files (default: {FRAME_ROOT})")
    args = ap.parse_args()

    h5_root     = Path(args.h5_root)     if args.h5_root     else H5_ROOT
    output_root = Path(args.output_root) if args.output_root else FRAME_ROOT

    print(f"Exporting source frames [{', '.join(args.norm)}]")
    print(f"  H5 root    : {h5_root}")
    print(f"  Output root: {output_root}\n")

    for ds in args.datasets:
        print(f"=== {ds} ===", flush=True)
        export_dataset(ds, args.norm, h5_root, output_root)

    print("\nAll done.", flush=True)


if __name__ == "__main__":
    main()
