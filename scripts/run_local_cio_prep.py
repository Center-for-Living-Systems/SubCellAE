#!/usr/bin/env python3
"""
run_local_cio_prep.py
=====================
Run CIO frame-extraction + patch-prep locally (no SLURM) for all four datasets,
then pack data.h5 files into the image_service output root.

Scale fix: both frameextract and patchprep use scale=5.0 so canvas images and
patches share the same CIO normalization (previously frameextract used 1.0 by
mistake — see context_viewer.md "Known issue" section).

Source CZI data:
  /mnt/p/Annabel/FA-ML/For-Liya_Data-Sets-that-look-good-and-contain-paxillin/

Output:
  Frames  : OUTPUT_ROOT/source_frames/{ds}/{cond}/{cond}_f{N:04d}_{ch}.tif
  Patches : OUTPUT_ROOT/{ds}/{cond}/tiff_patches32_mr10/...
  H5      : OUTPUT_ROOT/{ds}/data.h5

Usage:
    python scripts/run_local_cio_prep.py                      # all steps
    python scripts/run_local_cio_prep.py --steps frames       # frames only
    python scripts/run_local_cio_prep.py --steps patches      # patches only
    python scripts/run_local_cio_prep.py --steps pack         # pack H5 only
    python scripts/run_local_cio_prep.py --datasets vinc pfak # subset
    python scripts/run_local_cio_prep.py --conditions control # one condition
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import tifffile

# Make sure repo root is on sys.path
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from subcellae.pipeline.frameextract_pipeline import (
    ChannelExtractConfig, FrameExtractConfig, run_frameextract_pipeline,
)
from subcellae.pipeline.patchprep_pipeline import PipelineConfig, run_pipeline

# ── Paths ──────────────────────────────────────────────────────────────────────

DATA_SRC = Path(
    "/mnt/p/Annabel/FA-ML/"
    "For-Liya_Data-Sets-that-look-good-and-contain-paxillin"
)

OUTPUT_ROOT = Path("/mnt/p/image_service/data/FA_patch_data/cio")

CIO_SCALE   = 5.0    # both frameextract and patchprep — fixes the 5x mismatch
PAD_SIZE    = 64
PATCH_SIZE  = 32
MASK_RATIO  = 0.1    # mr10
PATCH_SUBDIR = "tiff_patches32_mr10"

# ── Dataset definitions ────────────────────────────────────────────────────────
# Each entry: (ds_name, date_folder, control_subfolder, ycomp_subfolder, channels)
# channels: list of (index, name)  — ch1 is always pax (segmentation channel)

_DS_DATE = {
    "vinc":   "20250311_eGFPZyxin488_Phalloidin405_Vinculin(rb)647_paxillin(m)568",
    "pfak":   "20250720_eGFP-Zyxin 488, Phalloidin 405, pFAK (rb) 647, paxillin(m)568/072025",
    "ppax":   "20250721_eGFP-Zyxin 488_Phalloidin405_pPaxy118(rb) 647_Pax(m)568",
    "nih3t3": "20260227_NIH3T3_ZyxinGFP,Phalloidin405,Vinc_rb647,Pax_m555_reduced_size_AH",
}

_DS_COND_FOLDERS = {
    #       control       ycomp
    "vinc":   ("Control",   "Ycomp"),
    "pfak":   ("Control",   "Ycomp"),
    "ppax":   ("Control",   "Y-comp"),
    "nih3t3": ("Control",   "YCompound"),
}

_DS_CHANNELS = {
    # ch index, ch name  — ch1 is always paxillin
    "vinc":   [(0, "vinc"), (1, "pax"), (2, "zyx"), (3, "act")],
    "pfak":   [(0, "pfak"), (1, "pax"), (2, "zyx"), (3, "act")],
    "ppax":   [(0, "ppax"), (1, "pax"), (2, "zyx"), (3, "act")],
    "nih3t3": [(0, "vinc"), (1, "pax"), (2, "zyx"), (3, "act")],
}

# Condition label used in filenames (must match patchprep convention)
_COND_LABEL = {
    "control": "control",
    "ycomp":   "ycomp",
}

ALL_DATASETS   = list(_DS_DATE.keys())
ALL_CONDITIONS = ["control", "ycomp"]

# Segmentation params (same as existing cluster configs)
_SEG = dict(
    seg_ch=1,
    seg_threshold=0.1,
    seg_close_size=11,
    seg_min_size_initial=3,
    seg_min_size_post_close=10,
    seg_min_size_final=30000,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _czi_dir(ds: str, cond: str) -> Path:
    """Return the directory containing raw CZI files for (ds, cond)."""
    cond_folder = _DS_COND_FOLDERS[ds][0 if cond == "control" else 1]
    return DATA_SRC / _DS_DATE[ds] / cond_folder


def _frames_dir(ds: str, cond: str) -> Path:
    return OUTPUT_ROOT / "source_frames" / ds / cond


def _patches_dir(ds: str, cond: str) -> Path:
    return OUTPUT_ROOT / ds / cond / PATCH_SUBDIR


def _plots_dir(ds: str, cond: str) -> Path:
    return OUTPUT_ROOT / ds / cond / "plot_patches32_mr10"


# ── Step 1: frameextract ──────────────────────────────────────────────────────

def run_frames(datasets: list[str], conditions: list[str]) -> None:
    """Extract CIO-normalised full frames (scale=5) from CZI files."""
    for ds in datasets:
        for cond in conditions:
            czi_dir = _czi_dir(ds, cond)
            if not czi_dir.exists():
                print(f"  [SKIP] frames {ds}/{cond}: {czi_dir} not found", flush=True)
                continue

            out_dir = _frames_dir(ds, cond)
            channels = [
                ChannelExtractConfig(index=idx, name=name, scale=CIO_SCALE)
                for idx, name in _DS_CHANNELS[ds]
            ]
            cfg = FrameExtractConfig(
                image_folder=str(czi_dir),
                output_dir=str(out_dir),
                condition=_COND_LABEL[cond],
                channels=channels,
                **_SEG,
            )
            print(f"\n=== frameextract {ds}/{cond} → {out_dir} ===", flush=True)
            run_frameextract_pipeline(cfg)


# ── Step 2: patchprep ─────────────────────────────────────────────────────────

def run_patches(datasets: list[str], conditions: list[str]) -> None:
    """Extract 32×32 CIO-normalised patches (scale=5) from CZI files."""
    for ds in datasets:
        for cond in conditions:
            czi_dir = _czi_dir(ds, cond)
            if not czi_dir.exists():
                print(f"  [SKIP] patches {ds}/{cond}: {czi_dir} not found", flush=True)
                continue

            cfg = PipelineConfig(
                image_folder=str(czi_dir),
                cell_mask_folder=None,
                movie_partitioned_data_dir=str(_patches_dir(ds, cond)),
                movie_plot_dir=str(_plots_dir(ds, cond)),
                condition=_COND_LABEL[cond],
                major_ch=1,
                patch_size=PATCH_SIZE,
                mask_ratio=MASK_RATIO,
                pad_size=PAD_SIZE,
                patch_prefix=_COND_LABEL[cond],
                norm_mode="cell_insideoutside",
                norm_cell_scale=CIO_SCALE,
                start_ind=0,
                end_ind=999,
                file_type="czi",
                **_SEG,
            )
            print(f"\n=== patchprep {ds}/{cond} → {_patches_dir(ds, cond)} ===", flush=True)
            run_pipeline(cfg)


# ── Step 3: pack data.h5 ──────────────────────────────────────────────────────

import re
_PATCH_RE = re.compile(r'^(.+)_f(\d+)x(\d+)y(\d+)ps(\d+)$')


def _parse_patch(stem: str):
    m = _PATCH_RE.match(stem)
    return (m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))) if m else None


def _load_frames_local(frame_dir: Path, cond: str, frame_indices: list[int],
                       channels: list[str]) -> dict[str, np.ndarray]:
    buf: dict[str, list] = {ch: [] for ch in channels}
    for fi in frame_indices:
        for ch in channels:
            p = frame_dir / f"{cond}_f{fi:04d}_{ch}.tif"
            buf[ch].append(tifffile.imread(str(p)).astype(np.float32) if p.exists() else None)

    result: dict[str, np.ndarray] = {}
    for ch in channels:
        frames = buf[ch]
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


def pack_dataset(ds: str, conditions: list[str]) -> Path | None:
    """Pack frames + patches for one dataset into data.h5."""
    out_path = OUTPUT_ROOT / ds / "data.h5"

    ds_ch_name = _DS_CHANNELS[ds][0][1]   # first channel name (vinc/pfak/ppax/vinc)
    channels = (
        ["pax", ds_ch_name, "zyx", "act"]
        if ds_ch_name not in ("pax", "zyx", "act")
        else ["pax", "zyx", "act"]
    )

    all_records:    list[dict]       = []
    all_patches:    list[np.ndarray] = []
    frame_arrays:   dict[str, list]  = {ch: [] for ch in channels}
    img_meta_rows:  list[dict]       = []
    global_frame_idx = 0

    for cond in conditions:
        patch_dir = _patches_dir(ds, cond)
        frame_dir = _frames_dir(ds, cond)

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
                "mean_intensity":        float(tifffile.imread(str(t)).mean()),
                "annotation_label":      -1,
                "annotation_label_name": "",
            })

        meta_df    = pd.DataFrame(records)
        frame_idxs = sorted(meta_df["frame_idx"].unique().tolist())
        fi_to_row  = {fi: global_frame_idx + i for i, fi in enumerate(frame_idxs)}
        meta_df["frame_row"] = meta_df["frame_idx"].map(fi_to_row)
        all_records.extend(records)

        patches = np.stack([tifffile.imread(str(t)).astype(np.float32) for t in tifs])
        all_patches.append(patches)
        print(f"    patches: {patches.shape}", flush=True)

        if frame_dir.exists():
            fa = _load_frames_local(frame_dir, cond, frame_idxs, channels)
            for ch in channels:
                if ch in fa:
                    for row in fa[ch]:
                        frame_arrays[ch].append(row)
                else:
                    ref_ch = next((c for c in channels if c in fa), None)
                    if ref_ch:
                        h, w = fa[ref_ch][0].shape
                        for _ in frame_idxs:
                            frame_arrays[ch].append(np.zeros((h, w), dtype=np.float32))
            for ch, arr in fa.items():
                print(f"    {ch} frames: {arr.shape}", flush=True)
        else:
            print(f"    WARNING: frame dir missing {frame_dir}", flush=True)

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

    meta_df   = pd.DataFrame(all_records)
    img_meta  = pd.DataFrame(img_meta_rows)
    patches_all = np.concatenate(all_patches, axis=0)
    print(f"\n  {ds}: total patches={len(patches_all)}, frames={global_frame_idx}", flush=True)

    (OUTPUT_ROOT / ds).mkdir(parents=True, exist_ok=True)
    with h5py.File(str(out_path), "w") as f:
        f.create_dataset("patches/raw", data=patches_all,
                         compression="gzip", compression_opts=4)

        if "pax" in frame_arrays and frame_arrays["pax"]:
            f.create_dataset("images/raw", data=np.stack(frame_arrays["pax"]),
                             compression="gzip", compression_opts=4)

        for ch in channels:
            if ch != "pax" and frame_arrays.get(ch):
                f.create_dataset(f"images/{ch}", data=np.stack(frame_arrays[ch]),
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
    return out_path


def run_pack(datasets: list[str], conditions: list[str]) -> None:
    for ds in datasets:
        print(f"\n=== pack {ds} ===", flush=True)
        pack_dataset(ds, conditions)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", nargs="+",
                    choices=["frames", "patches", "pack"],
                    default=["frames", "patches", "pack"],
                    help="Which steps to run (default: all three)")
    ap.add_argument("--datasets", nargs="+", default=ALL_DATASETS,
                    choices=ALL_DATASETS, metavar="DS")
    ap.add_argument("--conditions", nargs="+", default=ALL_CONDITIONS,
                    choices=ALL_CONDITIONS, metavar="COND")
    args = ap.parse_args()

    print(f"Steps     : {args.steps}")
    print(f"Datasets  : {args.datasets}")
    print(f"Conditions: {args.conditions}")
    print(f"CIO scale : {CIO_SCALE}  (applied to both frames and patches)")
    print(f"Output    : {OUTPUT_ROOT}\n")

    if "frames" in args.steps:
        print("── Step 1: frame extraction ──────────────────────────────", flush=True)
        run_frames(args.datasets, args.conditions)

    if "patches" in args.steps:
        print("\n── Step 2: patch extraction ──────────────────────────────", flush=True)
        run_patches(args.datasets, args.conditions)

    if "pack" in args.steps:
        print("\n── Step 3: pack data.h5 ──────────────────────────────────", flush=True)
        run_pack(args.datasets, args.conditions)

    print("\nAll done.", flush=True)


if __name__ == "__main__":
    main()
