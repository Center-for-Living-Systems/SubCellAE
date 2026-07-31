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

# Raw CZI folder per dataset/condition for robust re-normalization
_FA = DATA_ROOT / "fa_data/other_paxillin"
DS_RAW_FOLDERS: dict[str, dict[str, Path]] = {
    "vinc": {
        "control": _FA / "20250311_eGFPZyxin488_Phalloidin405_Vinculin(rb)647_paxillin(m)568/Control",
        "ycomp":   _FA / "20250311_eGFPZyxin488_Phalloidin405_Vinculin(rb)647_paxillin(m)568/Ycomp",
    },
    "pfak": {
        "control": _FA / "20250720_eGFP-Zyxin 488, Phalloidin 405, pFAK (rb) 647, paxillin(m)568/072025/Control",
        "ycomp":   _FA / "20250720_eGFP-Zyxin 488, Phalloidin 405, pFAK (rb) 647, paxillin(m)568/072025/Ycomp",
    },
    "ppax": {
        "control": _FA / "20250721_eGFP-Zyxin 488_Phalloidin405_pPaxy118(rb) 647_Pax(m)568/Control",
        "ycomp":   _FA / "20250721_eGFP-Zyxin 488_Phalloidin405_pPaxy118(rb) 647_Pax(m)568/Y-comp",
    },
    "nih3t3": {
        "control": _FA / "20260227_NIH3T3_ZyxinGFP,Phalloidin405,Vinc_rb647,Pax_m555_reduced_size_AH/Control",
        "ycomp":   _FA / "20260227_NIH3T3_ZyxinGFP,Phalloidin405,Vinc_rb647,Pax_m555_reduced_size_AH/YCompound",
    },
}
# All datasets share the same patchprep segmentation parameters
_SEG_PARAMS = dict(
    seg_ch=1, threshold=0.1, close_size=11,
    min_size_initial=3, min_size_post_close=10, min_size_final=30000,
)
# Rolling-ball radius: None for cio, 20 for cio_rb
_ROLLING_BALL = {"cio": None, "cio_rb": 20}

import re
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import subcellae.dataprep.patch_prep as _pp

_PATCH_RE = re.compile(r'^(.+)_f(\d+)x(\d+)y(\d+)ps(\d+)$')


def _parse_patch(stem: str):
    m = _PATCH_RE.match(stem)
    return (m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))) if m else None


def _compute_robust_patches(
    records: list[dict],
    ds: str,
    conditions: list[str],
    norm: str,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Re-normalize patches from raw CZI frames using trimmed-mean and median variants.

    Returns (patches_inlier, patches_med), each float32 (N, 32, 32), aligned with
    *records* order. Returns (None, None) if raw folders are unavailable.
    """
    if ds not in DS_RAW_FOLDERS:
        print(f"  [robust] no raw folder mapping for ds={ds!r}, skipping", flush=True)
        return None, None

    rolling_ball_radius = _ROLLING_BALL.get(norm)
    scale = 5.0  # matches patchprep default norm_cell_scale

    # index records by (condition, frame_idx) → list of (record_pos, x_c, y_c, ps)
    frame_map: dict[tuple[str, int], list[tuple[int, int, int, int]]] = defaultdict(list)
    for i, rec in enumerate(records):
        cond = rec["condition_name"]
        fi   = rec["frame_idx"]
        x_c  = rec["canvas_cx"] + PAD_SIZE
        y_c  = rec["canvas_cy"] + PAD_SIZE
        ps   = rec["ps"]
        frame_map[(cond, fi)].append((i, x_c, y_c, ps))

    patches_inlier = np.zeros((len(records), 32, 32), dtype=np.float32)
    patches_med    = np.zeros((len(records), 32, 32), dtype=np.float32)

    for cond in conditions:
        raw_folder = DS_RAW_FOLDERS[ds].get(cond)
        if raw_folder is None or not raw_folder.exists():
            print(f"  [robust] raw folder missing for {ds}/{cond}: {raw_folder}", flush=True)
            continue

        filenames = _pp.list_image_files(str(raw_folder), file_type="czi")
        if not filenames:
            print(f"  [robust] no CZI files in {raw_folder}", flush=True)
            continue

        # collect unique frame indices for this condition
        cond_frame_idxs = sorted({fi for (c, fi) in frame_map if c == cond})
        print(f"  [robust] {ds}/{cond}: re-normalising {len(cond_frame_idxs)} frames …", flush=True)

        for fi in cond_frame_idxs:
            if fi >= len(filenames):
                print(f"    WARNING: frame_idx={fi} out of range ({len(filenames)} files)", flush=True)
                continue

            filename = filenames[fi]
            raw = _pp._load_raw_squeezed(str(raw_folder), filename, "czi")

            # rolling ball on pax channel (ch1) before segmentation
            img = _pp._extract_channel(raw, 1, filename, "czi")
            if rolling_ball_radius is not None:
                _s = 255.0 * 255.0
                img = _pp.apply_rolling_ball(img * _s, radius=rolling_ball_radius) / _s

            seg_input = _pp._extract_channel(raw, _SEG_PARAMS["seg_ch"], filename, "czi")
            seg = _pp.segment_cell_mask(
                seg_input,
                threshold=_SEG_PARAMS["threshold"],
                close_size=_SEG_PARAMS["close_size"],
                min_size_initial=_SEG_PARAMS["min_size_initial"],
                min_size_post_close=_SEG_PARAMS["min_size_post_close"],
                min_size_final=_SEG_PARAMS["min_size_final"],
            ).astype(float)

            img_inlier = _pp.normalize_cell_insideoutside_inlier(img, seg, scale=scale)
            img_med    = _pp.normalize_cell_insideoutside_med(img, seg, scale=scale)

            img_inlier_pad = _pp.image_padding(img_inlier, PAD_SIZE, float(np.mean(img_inlier)))
            img_med_pad    = _pp.image_padding(img_med,    PAD_SIZE, float(np.mean(img_med)))

            for (rec_i, x_c, y_c, ps) in frame_map[(cond, fi)]:
                half = ps // 2
                r0, r1 = y_c - half, y_c + half
                c0, c1 = x_c - half, x_c + half
                patches_inlier[rec_i] = img_inlier_pad[r0:r1, c0:c1].astype(np.float32)
                patches_med[rec_i]    = img_med_pad[r0:r1, c0:c1].astype(np.float32)

    return patches_inlier, patches_med


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


def pack_dataset(ds: str, conditions: list[str], norm: str = "cio",
                 patches_root: Path | None = None,
                 frames_root:  Path | None = None) -> Path | None:
    if patches_root is None:
        patches_root = DATA_ROOT / f"ae_results/patches/{norm}"
    if frames_root is None:
        frames_root = DATA_ROOT / f"ae_results/source_frames/{norm}"
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
                "mean_intensity":        float(tifffile.imread(str(t)).mean()),
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

    # Robust normalization variants from raw frames
    patches_inlier, patches_med = _compute_robust_patches(
        all_records, ds, conditions, norm
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(out_path), "w") as f:
        f.create_dataset("patches/raw", data=patches_all,
                         compression="gzip", compression_opts=4)

        if patches_inlier is not None:
            f.create_dataset("patches/cio_inlier", data=patches_inlier,
                             compression="gzip", compression_opts=4)
            f.create_dataset("patches/cio_med", data=patches_med,
                             compression="gzip", compression_opts=4)
            print(f"  patches/cio_inlier and patches/cio_med saved", flush=True)

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
    ap.add_argument("--patches-root", default=None, metavar="PATH",
                    help="Override patches root dir (default: DATA_ROOT/ae_results/patches/{norm})")
    ap.add_argument("--frames-root", default=None, metavar="PATH",
                    help="Override frames root dir (default: DATA_ROOT/ae_results/source_frames/{norm})")
    args = ap.parse_args()

    pr = Path(args.patches_root) if args.patches_root else None
    fr = Path(args.frames_root)  if args.frames_root  else None

    print(f"Packing data.h5 [{args.norm}] for {args.datasets} × {args.conditions}\n", flush=True)
    if pr: print(f"  patches-root: {pr}")
    if fr: print(f"  frames-root : {fr}")
    for ds in args.datasets:
        print(f"=== {ds} ===", flush=True)
        pack_dataset(ds, args.conditions, norm=args.norm, patches_root=pr, frames_root=fr)

    print("\nAll done.", flush=True)


if __name__ == "__main__":
    main()
