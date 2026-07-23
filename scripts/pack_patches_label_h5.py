#!/usr/bin/env python3
"""
pack_patches_label_h5.py

Lightweight H5 packer for manual FA patch labelling.
No AE / UMAP / classifier outputs — just image + patch data.

Compatible with label_patches.py (same H5 layout as pack_interactive_h5.py).

Reads:
  ae_results/patches/cio_rb/{ds}/{cond}/tiff_patches32_label/*.tif
  ae_results/source_frames/cio_rb/{ds}/{cond}/{cond}_f{N}_{ch}.tif

Writes one H5 per (dataset, condition):
  ae_results/patches/cio_rb/{ds}/{ds}_{cond}_label.h5

H5 layout (label_patches.py compatible):
  patches/raw    float32 (N, 32, 32)  — patch images
  images/raw     float32 (M, H, W)   — paxillin frames (main canvas channel)
  images/act     float32 (M, H, W)   — actin frames
  images/{ds_ch} float32 (M, H, W)   — ds-specific marker (vinc/ppax/pfak)
  meta/csv       bytes (CSV)          — per-patch: filename, patch_group,
                                        condition_name, canvas_cx, canvas_cy, ps
  images/meta    bytes (CSV)          — per-frame: group, frame (row in images/raw)
attrs: pad_size=64, image_scale=1.0, dataset, condition, n_patches, n_frames

Usage:
    python scripts/pack_patches_label_h5.py
    python scripts/pack_patches_label_h5.py --datasets vinc pfak
    python scripts/pack_patches_label_h5.py --conditions control
"""
from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import tifffile

# ── config ─────────────────────────────────────────────────────────────────────

DATA_ROOT          = Path("/net/projects/CLS/lding/data/fa_data_analysis")
IMAGE_SERVICE_ROOT = Path("/mnt/p/image_service/data/FA_patch_data/cio_rb")
PATCHES_ROOT = DATA_ROOT / "ae_results/patches/cio_rb"
FRAMES_ROOT  = DATA_ROOT / "ae_results/source_frames/cio_rb"

# Dataset-specific FA marker channel name in source_frames filenames
DS_CHANNEL = {
    "vinc":  "vinc",
    "nih3t3": "vinc",   # same CZI layout as vinc
    "ppax":  "ppax",
    "pfak":  "pfak",
}

ALL_DATASETS   = list(DS_CHANNEL.keys())
ALL_CONDITIONS = ["control", "ycomp"]

PAD_SIZE = 64  # matches patchprep pad_size in all label configs

# Regex: {cond}_f{frame}x{x}y{y}ps{ps}.tif
# x, y are CENTER coordinates in the padded image space
_PATCH_RE = re.compile(r'^(.+)_f(\d+)x(\d+)y(\d+)ps(\d+)$')


def _parse_patch_name(stem: str):
    """Return (cond_prefix, frame_idx, x_center, y_center, ps) or None."""
    m = _PATCH_RE.match(stem)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
    return None


def _load_frames(frame_dir: Path, cond: str, frame_indices: list[int],
                 channels: list[str]) -> dict[str, np.ndarray]:
    """Load selected source frames for each channel. Returns {ch: (M, H, W) array}."""
    arrays: dict[str, list] = {ch: [] for ch in channels}
    missing_frames = []
    for fi in frame_indices:
        for ch in channels:
            p = frame_dir / f"{cond}_f{fi:04d}_{ch}.tif"
            if p.exists():
                arrays[ch].append(tifffile.imread(str(p)).astype(np.float32))
            else:
                arrays[ch].append(None)
                if ch == channels[0]:
                    missing_frames.append(fi)

    result = {}
    for ch in channels:
        frames = arrays[ch]
        valid = [f for f in frames if f is not None]
        if not valid:
            print(f"    WARNING: no '{ch}' frames found", flush=True)
            continue
        h, w = valid[0].shape[-2], valid[0].shape[-1]
        stack = np.zeros((len(frames), h, w), dtype=np.float32)
        for i, f in enumerate(frames):
            if f is not None:
                stack[i] = f if f.ndim == 2 else f[0]
        result[ch] = stack

    return result


def pack(ds: str, cond: str) -> Path | None:
    patch_dir = PATCHES_ROOT / ds / cond / "tiff_patches32_label"
    frame_dir = FRAMES_ROOT / ds / cond
    out_dir   = PATCHES_ROOT / ds
    out_path  = out_dir / f"{ds}_{cond}_label.h5"

    if not patch_dir.exists():
        print(f"  SKIP {ds}/{cond}: patch dir not found: {patch_dir}", flush=True)
        return None

    tifs = sorted(patch_dir.glob("*.tif"))
    if not tifs:
        print(f"  SKIP {ds}/{cond}: no patches in {patch_dir}", flush=True)
        return None

    print(f"  {ds}/{cond}: {len(tifs)} patches …", flush=True)

    # ── parse patch metadata ─────────────────────────────────────────────────
    records = []
    for t in tifs:
        parsed = _parse_patch_name(t.stem)
        if parsed is None:
            print(f"    WARNING: unrecognised filename: {t.name}")
            continue
        prefix, frame_idx, x, y, ps = parsed
        group = f"{cond}_f{frame_idx:04d}"
        records.append({
            "filename":       t.name,
            "patch_group":    group,
            "condition_name": cond,
            "frame_idx":      frame_idx,
            # x, y are center coords in padded space; subtract pad_size for canvas
            "canvas_cx":      x - PAD_SIZE,
            "canvas_cy":      y - PAD_SIZE,
            "ps":             ps,
        })

    meta_df = pd.DataFrame(records)
    frame_indices = sorted(meta_df["frame_idx"].unique().tolist())
    frame_to_row  = {fi: i for i, fi in enumerate(frame_indices)}

    print(f"    frames: {len(frame_indices)}  unique", flush=True)

    # ── load patches ─────────────────────────────────────────────────────────
    patches = np.stack([tifffile.imread(str(t)).astype(np.float32) for t in tifs], axis=0)
    print(f"    patches loaded: {patches.shape}", flush=True)

    # ── load source frames ───────────────────────────────────────────────────
    ds_ch    = DS_CHANNEL[ds]
    # Order: pax (main canvas), then marker, zyxin, actin — so side panels show [marker, zyx, act]
    channels = (["pax", ds_ch, "zyx", "act"] if ds_ch not in ("pax", "zyx", "act")
                else ["pax", "zyx", "act"])
    if not frame_dir.exists():
        print(f"    WARNING: frame dir missing: {frame_dir}", flush=True)
        frame_arrays = {}
    else:
        frame_arrays = _load_frames(frame_dir, cond, frame_indices, channels)
        for ch, arr in frame_arrays.items():
            print(f"    {ch} frames: {arr.shape}", flush=True)

    # ── image metadata (label_patches.py compatible) ─────────────────────────
    img_meta = pd.DataFrame({
        "group": [f"{cond}_f{fi:04d}" for fi in frame_indices],
        "frame": list(range(len(frame_indices))),   # row index into images/raw
        "frame_idx": frame_indices,
    })

    # ── write H5 (label_patches.py compatible layout) ────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(out_path), "w") as f:
        f.create_dataset("patches/raw", data=patches,
                         compression="gzip", compression_opts=4)

        # pax is the main canvas channel expected by label_patches.py
        if "pax" in frame_arrays:
            f.create_dataset("images/raw", data=frame_arrays["pax"],
                             compression="gzip", compression_opts=4)
        # keep additional channels alongside
        for ch, arr in frame_arrays.items():
            if ch != "pax":
                f.create_dataset(f"images/{ch}", data=arr,
                                 compression="gzip", compression_opts=4)

        # label_patches.py reads these with f[key][()].decode() — must be scalar bytes
        f.create_dataset("meta/csv",    data=np.bytes_(meta_df.to_csv(index=False)))
        f.create_dataset("images/meta", data=np.bytes_(img_meta.to_csv(index=False)))

        # required attrs
        f.attrs["pad_size"]    = float(PAD_SIZE)
        f.attrs["image_scale"] = 1.0
        f.attrs["result_dir"]  = ""
        f.attrs["dataset"]     = ds
        f.attrs["condition"]   = cond
        f.attrs["n_patches"]   = len(patches)
        f.attrs["n_frames"]    = len(frame_indices)
        f.attrs["channels"]    = list(frame_arrays.keys())

    size_mb = out_path.stat().st_size / 1e6
    print(f"    → {out_path}  ({size_mb:.1f} MB)", flush=True)

    # Copy to image_service NAS if mounted
    if IMAGE_SERVICE_ROOT.exists():
        import shutil
        svc_dir  = IMAGE_SERVICE_ROOT / ds
        svc_path = svc_dir / f"{ds}_{cond}_label.h5"
        try:
            svc_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(out_path), str(svc_path))
            print(f"    → {svc_path}  (image_service copy)", flush=True)
        except Exception as exc:
            print(f"    [warn] image_service copy failed: {exc}", flush=True)
    else:
        print(f"    [skip] image_service not mounted ({IMAGE_SERVICE_ROOT})", flush=True)

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Pack label H5 files (patches + images).")
    parser.add_argument("--datasets",   nargs="+", default=ALL_DATASETS,
                        choices=ALL_DATASETS, metavar="DS",
                        help="Datasets to pack (default: all)")
    parser.add_argument("--conditions", nargs="+", default=ALL_CONDITIONS,
                        choices=ALL_CONDITIONS, metavar="COND",
                        help="Conditions to pack (default: control ycomp)")
    args = parser.parse_args()

    print(f"Packing {args.datasets} × {args.conditions}", flush=True)
    for ds in args.datasets:
        for cond in args.conditions:
            pack(ds, cond)

    print("\nAll done.", flush=True)


if __name__ == "__main__":
    main()
