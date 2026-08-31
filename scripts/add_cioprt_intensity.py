#!/usr/bin/env python3
"""
add_cioprt_intensity.py

For each patch in the existing cellprofiler feature CSV, load the corresponding
cio_mode_prt source frame, extract the same patch region, compute 11 intensity
features, and append them as extra columns (suffix _cioprt).

Patch filename format: {cond}_f{frame_idx:04d}x{x:04d}y{y:04d}ps{ps}.tif
Source frame path:     source_frames/cio_mode_prt/{ds_name}/{cond}/{cond}_f{frame_idx:04d}_pax.tif

Usage
-----
  python scripts/add_cioprt_intensity.py --dataset ds1
  python scripts/add_cioprt_intensity.py --all
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

DATA       = Path("/net/projects/CLS/lding/data/fa_data_analysis")
FEAT_DIR   = DATA / "ae_results/features/cellprofiler"
FRAME_ROOT = DATA / "ae_results/source_frames/cio_mode_prt"

# Maps ds key → ds_name in the source frame directory
DS_NAME = {"ds1": "vinc", "ds2": "pfak", "ds3": "ppax"}

# Maps ds key → list of (ds_name, cond) pairs matching PATCH_DIRS order
DS_CONDS = {
    "ds1": [("vinc", "control"), ("vinc", "ycomp")],
    "ds2": [("pfak", "control"), ("pfak", "ycomp")],
    "ds3": [("ppax", "control"), ("ppax", "ycomp")],
}

PATCH_RE = re.compile(r"^(.+)_f(\d+)x(\d+)y(\d+)ps(\d+)\.tif$")

INTENSITY_NAMES_CIOPRT = [f"{n}_cioprt" for n in [
    "intensity_mean", "intensity_std", "intensity_median",
    "intensity_min",  "intensity_max", "intensity_integrated",
    "intensity_mad",  "intensity_p10", "intensity_p25",
    "intensity_p75",  "intensity_p90",
]]


def intensity_features(img: np.ndarray) -> np.ndarray:
    flat = img.ravel()
    mad  = np.median(np.abs(flat - np.median(flat)))
    return np.array([
        flat.mean(), flat.std(), np.median(flat),
        flat.min(), flat.max(), flat.sum(), mad,
        np.percentile(flat, 10), np.percentile(flat, 25),
        np.percentile(flat, 75), np.percentile(flat, 90),
    ], dtype=np.float32)


def process_dataset(ds: str):
    csv_path = FEAT_DIR / f"{ds}.csv"
    df = pd.read_csv(csv_path)
    # Drop any existing _cioprt columns from a previous run
    drop_cols = [c for c in df.columns if c.endswith("_cioprt")]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    print(f"{ds}: {len(df)} patches")

    # Build frame cache: (ds_name, cond, frame_idx) → frame array
    frame_cache: dict[tuple, np.ndarray] = {}

    def _load_frame(ds_name: str, cond: str, frame_idx: int) -> np.ndarray | None:
        key = (ds_name, cond, frame_idx)
        if key not in frame_cache:
            p = FRAME_ROOT / ds_name / cond / f"{cond}_f{frame_idx:04d}_pax.tif"
            if not p.exists():
                return None
            frame_cache[key] = tifffile.imread(str(p)).astype(np.float32)
        return frame_cache[key]

    results = []
    n_miss = 0
    for i, row in df.iterrows():
        if i % 2000 == 0:
            print(f"  {i}/{len(df)} ...", flush=True)

        fn = row["filename"]
        m  = PATCH_RE.match(fn)
        if not m:
            results.append([np.nan] * 11)
            n_miss += 1
            continue

        cond_name = m.group(1)      # e.g. "control" or "ycomp"
        frame_idx = int(m.group(2))
        x_left    = int(m.group(3))
        y_left    = int(m.group(4))
        ps        = int(m.group(5))

        # Find ds_name for this cond
        ds_name = None
        for dn, cn in DS_CONDS[ds]:
            if cn == cond_name:
                ds_name = dn
                break
        if ds_name is None:
            results.append([np.nan] * 11)
            n_miss += 1
            continue

        frame = _load_frame(ds_name, cond_name, frame_idx)
        if frame is None:
            results.append([np.nan] * 11)
            n_miss += 1
            continue

        raw   = frame[y_left:y_left + ps, x_left:x_left + ps]
        if raw.shape != (ps, ps):
            if raw.size == 0:
                # fully out of bounds
                results.append([np.nan] * 11)
                n_miss += 1
                continue
            # partial edge patch — pad with edge values
            ph = ps - raw.shape[0]
            pw = ps - raw.shape[1]
            raw = np.pad(raw, ((0, ph), (0, pw)), mode="edge")
        patch = raw

        results.append(intensity_features(patch).tolist())

    feat_df = pd.DataFrame(results, columns=INTENSITY_NAMES_CIOPRT)
    out_df  = pd.concat([df, feat_df], axis=1)
    out_df.to_csv(csv_path, index=False)
    print(f"  Saved {len(out_df)} rows, {n_miss} missing → {csv_path}")
    frame_cache.clear()


def main():
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--dataset", choices=["ds1", "ds2", "ds3"])
    grp.add_argument("--all", action="store_true")
    args = ap.parse_args()

    datasets = ["ds1", "ds2", "ds3"] if args.all else [args.dataset]
    for ds in datasets:
        process_dataset(ds)
    print("Done.")


if __name__ == "__main__":
    main()
