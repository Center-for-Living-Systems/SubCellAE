#!/usr/bin/env python3
"""
pack_data_h5.py
================
Build one shared data.h5 per dataset (all conditions combined).

This is the static half of the two-file H5 design:
  data.h5   ← this script  (images + patches, shared across models)
  model.h5  ← pack_model_h5.py  (latents, UMAP, recon, predictions)

Sources (cio_rb, mr10 patchprep):
  ae_results/patches/cio_rb/{ds}/{cond}/tiff_patches32_mr10/*.tif
  ae_results/source_frames/cio_rb/{ds}/{cond}/{cond}_f{N}_{ch}.tif

Output:
  ae_results/patches/cio_rb/{ds}/data.h5

H5 layout
---------
  patches/raw       float32 (N, 32, 32)   — all conditions combined
  images/raw        float32 (M, H, W)     — paxillin frames, all conditions
  images/{ch}       float32 (M, H, W)     — one dataset per additional channel
  images/meta       bytes (CSV)           — group, frame (row in images/raw),
                                            condition_name, frame_idx
  meta/csv          bytes (CSV)           — per-patch static metadata:
                                            filename, condition_name, group,
                                            frame_idx, canvas_cx, canvas_cy, ps,
                                            mean_intensity, annotation_label,
                                            annotation_label_name
  attrs: pad_size, image_scale, dataset, n_patches, n_frames, channels (JSON)

Usage:
    python scripts/pack_data_h5.py                     # all 4 datasets
    python scripts/pack_data_h5.py --datasets vinc pfak
    python scripts/pack_data_h5.py --conditions control
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import tifffile

DATA_ROOT          = Path("/net/projects/CLS/lding/data/fa_data_analysis")
IMAGE_SERVICE_ROOT = Path("/mnt/p/image_service/data/FA_patch_data")
PATCH_SUBDIR = "tiff_patches32_mr10"
PAD_SIZE     = 64   # patchprep pad_size for mr10 configs

DS_CHANNEL = {
    "vinc":   "vinc",
    "nih3t3": "vinc",
    "ppax":   "ppax",
    "pfak":   "pfak",
}
ALL_DATASETS   = list(DS_CHANNEL.keys())
ALL_CONDITIONS = ["control", "ycomp"]
ALL_NORMS      = ["cio", "cio_rb"]

import re
_PATCH_RE = re.compile(r'^(.+)_f(\d+)x(\d+)y(\d+)ps(\d+)$')


def _parse_patch(stem: str):
    m = _PATCH_RE.match(stem)
    return (m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))) if m else None


def _load_frames(frame_dir: Path, cond: str, frame_indices: list[int],
                 channels: list[str]) -> dict[str, np.ndarray]:
    """Load selected source frames per channel → {ch: (M, H, W) float32}."""
    buf: dict[str, list] = {ch: [] for ch in channels}
    for fi in frame_indices:
        for ch in channels:
            p = frame_dir / f"{cond}_f{fi:04d}_{ch}.tif"
            buf[ch].append(tifffile.imread(str(p)).astype(np.float32) if p.exists() else None)

    result: dict[str, np.ndarray] = {}
    for ch in channels:
        frames = buf[ch]
        valid  = [f for f in frames if f is not None]
        if not valid:
            print(f"    WARNING: no '{ch}' frames found", flush=True)
            continue
        h, w   = valid[0].shape[-2], valid[0].shape[-1]
        stack  = np.zeros((len(frames), h, w), dtype=np.float32)
        for i, f in enumerate(frames):
            if f is not None:
                stack[i] = f if f.ndim == 2 else f[0]
        result[ch] = stack
    return result


def pack_dataset(ds: str, conditions: list[str], norm: str = "cio") -> Path | None:
    patches_root = DATA_ROOT / f"ae_results/patches/{norm}"
    frames_root  = DATA_ROOT / f"ae_results/source_frames/{norm}"
    out_dir  = patches_root / ds
    out_path = out_dir / "data.h5"

    all_records:  list[dict]      = []
    all_patches:  list[np.ndarray] = []
    frame_arrays: dict[str, list[np.ndarray]] = {}  # ch → list of (H,W) arrays
    img_meta_rows: list[dict]     = []
    global_frame_idx = 0  # row counter in images/raw across all conditions

    ds_ch    = DS_CHANNEL[ds]
    channels = (["pax", ds_ch, "zyx", "act"] if ds_ch not in ("pax", "zyx", "act")
                else ["pax", "zyx", "act"])
    for ch in channels:
        frame_arrays[ch] = []

    for cond in conditions:
        patch_dir = patches_root / ds / cond / PATCH_SUBDIR
        frame_dir = frames_root  / ds / cond

        if not patch_dir.exists():
            print(f"  SKIP {ds}/{cond}: {patch_dir} not found", flush=True)
            continue

        tifs = sorted(patch_dir.glob("*.tif"))
        if not tifs:
            print(f"  SKIP {ds}/{cond}: no patches", flush=True)
            continue

        print(f"  {ds}/{cond}: {len(tifs)} patches …", flush=True)

        records: list[dict] = []
        for t in tifs:
            parsed = _parse_patch(t.stem)
            if parsed is None:
                print(f"    WARNING: unrecognised filename: {t.name}")
                continue
            prefix, fi, x, y, ps = parsed
            records.append({
                "filename":              t.name,
                "condition_name":        cond,
                "group":                 f"{cond}_f{fi:04d}",
                "frame_idx":             fi,
                "canvas_cx":             x - PAD_SIZE,
                "canvas_cy":             y - PAD_SIZE,
                "ps":                    ps,
                "mean_intensity":        float((_arr := tifffile.imread(str(t)).astype(np.float32)).mean()),
                "max_intensity":         float(_arr.max()),
                "annotation_label":      -1,
                "annotation_label_name": "",
            })

        meta_df     = pd.DataFrame(records)
        frame_idxs  = sorted(meta_df["frame_idx"].unique().tolist())
        fi_to_row   = {fi: global_frame_idx + i for i, fi in enumerate(frame_idxs)}

        # update group row references
        meta_df["frame_row"] = meta_df["frame_idx"].map(fi_to_row)

        all_records.extend(records)

        # patches
        patches = np.stack([tifffile.imread(str(t)).astype(np.float32) for t in tifs])
        all_patches.append(patches)
        print(f"    patches: {patches.shape}", flush=True)

        # source frames
        if frame_dir.exists():
            fa = _load_frames(frame_dir, cond, frame_idxs, channels)
            for ch in channels:
                if ch in fa:
                    for row in fa[ch]:
                        frame_arrays[ch].append(row)
                else:
                    # pad with zeros so all channels stay aligned
                    ref_ch = next((c for c in channels if c in fa), None)
                    if ref_ch:
                        h, w = fa[ref_ch][0].shape
                        for _ in frame_idxs:
                            frame_arrays[ch].append(np.zeros((h, w), dtype=np.float32))
            for ch, arr in fa.items():
                print(f"    {ch} frames: {arr.shape}", flush=True)
        else:
            print(f"    WARNING: frame dir missing {frame_dir}", flush=True)

        # image meta rows
        for fi in frame_idxs:
            img_meta_rows.append({
                "group":          f"{cond}_f{fi:04d}",
                "frame":          fi_to_row[fi],
                "condition_name": cond,
                "frame_idx":      fi,
            })

        global_frame_idx += len(frame_idxs)

    if not all_records:
        print(f"  SKIP {ds}: no data found", flush=True)
        return None

    meta_df    = pd.DataFrame(all_records)
    img_meta   = pd.DataFrame(img_meta_rows)
    patches_all = np.concatenate(all_patches, axis=0)
    print(f"\n  {ds}: total patches={len(patches_all)}, frames={global_frame_idx}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(out_path), "w") as f:
        f.create_dataset("patches/raw", data=patches_all,
                         compression="gzip", compression_opts=4)

        if "pax" in frame_arrays and frame_arrays["pax"]:
            stack = np.stack(frame_arrays["pax"])
            f.create_dataset("images/raw", data=stack,
                             compression="gzip", compression_opts=4)

        for ch in channels:
            if ch != "pax" and frame_arrays.get(ch):
                stack = np.stack(frame_arrays[ch])
                f.create_dataset(f"images/{ch}", data=stack,
                                 compression="gzip", compression_opts=4)

        f.create_dataset("meta/csv",    data=np.bytes_(meta_df.to_csv(index=False)))
        f.create_dataset("images/meta", data=np.bytes_(img_meta.to_csv(index=False)))

        f.attrs["pad_size"]    = float(PAD_SIZE)
        f.attrs["image_scale"] = 1.0
        f.attrs["dataset"]     = ds
        f.attrs["n_patches"]   = int(len(patches_all))
        f.attrs["n_frames"]    = int(global_frame_idx)
        f.attrs["channels"]    = json.dumps(channels)

    size_mb = out_path.stat().st_size / 1e6
    print(f"  → {out_path}  ({size_mb:.1f} MB)", flush=True)

    # Copy to image_service NAS if mounted
    svc_dir = IMAGE_SERVICE_ROOT / norm / ds
    if IMAGE_SERVICE_ROOT.exists():
        import shutil
        try:
            svc_dir.mkdir(parents=True, exist_ok=True)
            svc_path = svc_dir / "data.h5"
            shutil.copy2(str(out_path), str(svc_path))
            print(f"  → {svc_path}  (image_service copy)", flush=True)
        except Exception as exc:
            print(f"  [warn] image_service copy failed: {exc}", flush=True)
    else:
        print(f"  [skip] image_service not mounted ({IMAGE_SERVICE_ROOT})", flush=True)

    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets",   nargs="+", default=ALL_DATASETS,
                    choices=ALL_DATASETS, metavar="DS")
    ap.add_argument("--conditions", nargs="+", default=ALL_CONDITIONS,
                    choices=ALL_CONDITIONS, metavar="COND")
    ap.add_argument("--norm", default="cio", choices=ALL_NORMS,
                    help="Normalisation variant to pack (default: cio)")
    args = ap.parse_args()

    print(f"Packing data.h5 [{args.norm}] for {args.datasets} × {args.conditions}\n", flush=True)
    for ds in args.datasets:
        print(f"=== {ds} ===", flush=True)
        pack_dataset(ds, args.conditions, norm=args.norm)

    print("\nAll done.", flush=True)


if __name__ == "__main__":
    main()
