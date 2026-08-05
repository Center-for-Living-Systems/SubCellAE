#!/usr/bin/env python3
"""
pack_data_h5.py
================
Build one shared data.h5 per dataset (all conditions combined).

This is the static half of the two-file H5 design:
  data.h5   ← this script  (images + patches, shared across models)
  model.h5  ← pack_model_h5.py  (latents, UMAP, recon, predictions)

Canvas images are generated from raw CZI files using the same cell-based
normalisation as the patches, so that patch pixel values and canvas pixel
values are on the same scale.

Sources:
  Raw CZI files  → normalise → patches/cio_inlier, patches/cio_med, images/*
  Pre-computed TIFFs (tiff_patches32_mr10) → patches/raw

CZI channel order (all four datasets share this layout):
  ch0 : marker channel  (vinculin / pPax / pFAK)
  ch1 : paxillin        (used for segmentation)
  ch2 : zyxin
  ch3 : actin

Output:
  {patches_root}/{ds}/data.h5

H5 layout
---------
  patches/raw        float32 (N, 32, 32)  — pre-computed patchprep TIFFs
  patches/cio_inlier float32 (N, 32, 32)  — inlier-normed from raw CZI
  patches/cio_med    float32 (N, 32, 32)  — median-normed from raw CZI
  images/raw         float32 (M, H, W)    — paxillin canvas (inlier-normed)
  images/{ch}        float32 (M, H, W)    — other channels (inlier-normed)
  images/meta        bytes (CSV)          — group, frame, condition_name, frame_idx
  meta/csv           bytes (CSV)          — per-patch metadata
  attrs: pad_size, image_scale, dataset, n_patches, n_frames, channels (JSON)

Usage:
    python scripts/pack_data_h5.py                     # all 4 datasets
    python scripts/pack_data_h5.py --datasets vinc pfak
    python scripts/pack_data_h5.py --norm cio_rb
    python scripts/pack_data_h5.py --raw-data-root /mnt/p/Annabel/FA-ML/...

    # Use pre-detected TIF patches from the CIO run, re-normalise into cio_inlier_med:
    python scripts/pack_data_h5.py \
        --patches-root /home/.../ae_results/patches/cio_inlier_med \
        --tif-patches-root /mnt/p/image_service/data/FA_patch_data/cio \
        --raw-data-root /mnt/p/Annabel/FA-ML/For-Liya_Data-Sets-that-look-good-and-contain-paxillin
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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
ALL_NORMS      = ["cio", "cio_rb", "cio_inlier_med"]

# CZI channel index for each biological channel (consistent across all 4 datasets)
_CZI_CH = {'pax': 1, 'zyx': 2, 'act': 3, 'vinc': 0, 'ppax': 0, 'pfak': 0}

# Rolling-ball radius: None for cio/cio_inlier_med, 20 for cio_rb
_ROLLING_BALL = {"cio": None, "cio_rb": 20, "cio_inlier_med": None}

NORM_KEYS = ["cio_inlier", "cio_med", "cio_mode", "cio_mode_prt"]

# ── Raw CZI folder maps ───────────────────────────────────────────────────────

_CLUSTER_FA = DATA_ROOT / "fa_data/other_paxillin"
DS_RAW_FOLDERS: dict[str, dict[str, Path]] = {
    "vinc": {
        "control": _CLUSTER_FA / "20250311_eGFPZyxin488_Phalloidin405_Vinculin(rb)647_paxillin(m)568/Control",
        "ycomp":   _CLUSTER_FA / "20250311_eGFPZyxin488_Phalloidin405_Vinculin(rb)647_paxillin(m)568/Ycomp",
    },
    "pfak": {
        "control": _CLUSTER_FA / "20250720_eGFP-Zyxin 488, Phalloidin 405, pFAK (rb) 647, paxillin(m)568/072025/Control",
        "ycomp":   _CLUSTER_FA / "20250720_eGFP-Zyxin 488, Phalloidin 405, pFAK (rb) 647, paxillin(m)568/072025/Ycomp",
    },
    "ppax": {
        "control": _CLUSTER_FA / "20250721_eGFP-Zyxin 488_Phalloidin405_pPaxy118(rb) 647_Pax(m)568/Control",
        "ycomp":   _CLUSTER_FA / "20250721_eGFP-Zyxin 488_Phalloidin405_pPaxy118(rb) 647_Pax(m)568/Y-comp",
    },
    "nih3t3": {
        "control": _CLUSTER_FA / "20260227_NIH3T3_ZyxinGFP,Phalloidin405,Vinc_rb647,Pax_m555_reduced_size_AH/Control",
        "ycomp":   _CLUSTER_FA / "20260227_NIH3T3_ZyxinGFP,Phalloidin405,Vinc_rb647,Pax_m555_reduced_size_AH/YCompound",
    },
}

_LOCAL_FA = Path("/mnt/p/Annabel/FA-ML/For-Liya_Data-Sets-that-look-good-and-contain-paxillin")
DS_RAW_FOLDERS_LOCAL: dict[str, dict[str, Path]] = {
    "vinc": {
        "control": _LOCAL_FA / "20250311_eGFPZyxin488_Phalloidin405_Vinculin(rb)647_paxillin(m)568/Control",
        "ycomp":   _LOCAL_FA / "20250311_eGFPZyxin488_Phalloidin405_Vinculin(rb)647_paxillin(m)568/Ycomp",
    },
    "pfak": {
        "control": _LOCAL_FA / "20250720_eGFP-Zyxin 488, Phalloidin 405, pFAK (rb) 647, paxillin(m)568/072025/Control",
        "ycomp":   _LOCAL_FA / "20250720_eGFP-Zyxin 488, Phalloidin 405, pFAK (rb) 647, paxillin(m)568/072025/Ycomp",
    },
    "ppax": {
        "control": _LOCAL_FA / "20250721_eGFP-Zyxin 488_Phalloidin405_pPaxy118(rb) 647_Pax(m)568/Control",
        "ycomp":   _LOCAL_FA / "20250721_eGFP-Zyxin 488_Phalloidin405_pPaxy118(rb) 647_Pax(m)568/Y-comp",
    },
    "nih3t3": {
        "control": _LOCAL_FA / "20260227_NIH3T3_ZyxinGFP,Phalloidin405,Vinc_rb647,Pax_m555_reduced_size_AH/Control",
        "ycomp":   _LOCAL_FA / "20260227_NIH3T3_ZyxinGFP,Phalloidin405,Vinc_rb647,Pax_m555_reduced_size_AH/YCompound",
    },
}

_SEG_PARAMS = dict(
    seg_ch=1, threshold=0.1, close_size=11,
    min_size_initial=3, min_size_post_close=10, min_size_final=30000,
)

import re
import scipy.ndimage as _ndi
from skimage.morphology import (binary_closing, binary_opening, disk,
                                remove_small_objects)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import subcellae.dataprep.patch_prep as _pp

_PATCH_RE = re.compile(r'^(.+)_f(\d+)x(\d+)y(\d+)ps(\d+)$')


def _segment_exclude_mask(pax_raw: np.ndarray,
                          min_size_final: int = 1000) -> np.ndarray:
    """Return bool mask of ALL cell-like regions for background exclusion.

    Mirrors segment_cell_mask steps 0-8 with a lower min_size_final, but
    skips step 9 (centre-region filter) so partial cells at image borders
    are also masked out and do not contaminate background statistics.
    """
    img = _pp._correct_seg_illumination(pax_raw)
    norm = _pp._percentile_stretch(img,
                                   p1=float(np.percentile(img, 1)),
                                   p99=float(np.percentile(img, 99)))
    mask = norm > _SEG_PARAMS["threshold"]
    mask = remove_small_objects(mask, min_size=_SEG_PARAMS["min_size_initial"], connectivity=1)
    mask = binary_closing(mask, disk(_SEG_PARAMS["close_size"]))
    mask = remove_small_objects(mask, min_size=_SEG_PARAMS["min_size_post_close"], connectivity=1)
    mask = binary_opening(mask, disk(3))
    mask = _ndi.binary_fill_holes(mask)
    mask = remove_small_objects(mask, min_size=min_size_final, connectivity=1)
    return mask.astype(bool)  # all regions kept — no centre filter


def _compute_channel_norms(ch_raw: np.ndarray,
                            bg_px: np.ndarray,
                            cell_px: np.ndarray,
                            scale: float = 5.0,
                            label: str = "") -> dict[str, np.ndarray]:
    """All 4 CIO normalizations for one channel array."""
    def _trimmed_mean(px: np.ndarray) -> float:
        if not px.size: return 0.0
        lo, hi = np.percentile(px, [1, 99])
        keep = px[(px >= lo) & (px <= hi)]
        return float(np.mean(keep)) if len(keep) else float(np.mean(px))

    def _mode16(px: np.ndarray) -> float:
        if not px.size: return 0.0
        counts = np.clip(np.round(px * 65536).astype(int), 0, 65535)
        return float(np.bincount(counts).argmax()) / 65536.0

    m_bg_in    = _trimmed_mean(bg_px)
    m_cell_in  = _trimmed_mean(cell_px)
    m_bg_med   = float(np.median(bg_px))   if bg_px.size  else 0.0
    m_cell_med = float(np.median(cell_px)) if cell_px.size else 0.0
    m_bg_mode  = _mode16(bg_px)

    if cell_px.size:
        _p975, _p995 = np.percentile(cell_px, [97.5, 99.5])
        _keep = cell_px[(_p975 < cell_px) & (cell_px < _p995)]
        if not len(_keep):
            # Saturation: flat top band, walk down percentile bands
            for _lo_pct, _hi_pct in [(95, 97.5), (92.5, 95), (90, 92.5), (85, 90), (80, 85)]:
                _plo, _phi = np.percentile(cell_px, [_lo_pct, _hi_pct])
                _keep = cell_px[(_plo < cell_px) & (cell_px <= _phi)]
                if len(_keep):
                    tag = label + " " if label else ""
                    print(f"  [cio_mode_prt] {tag}saturated P97.5=P99.5={_p995*65536:.0f}; "
                          f"using P{_lo_pct}–P{_hi_pct} band (mean={np.mean(_keep)*65536:.0f})", flush=True)
                    break
        m_cell_prt = float(np.mean(_keep)) if len(_keep) else m_cell_in
    else:
        m_cell_prt = m_bg_mode + 1.0

    raw = ch_raw.astype(np.float32)

    def _n(bg: float, cell_ref: float, s: float = scale) -> np.ndarray:
        d = (cell_ref - bg) * s or 1.0
        return (raw - bg) / d

    return {
        "cio_inlier":   _n(m_bg_in,   m_cell_in),
        "cio_med":      _n(m_bg_med,  m_cell_med),
        "cio_mode":     _n(m_bg_mode, m_cell_in),
        "cio_mode_prt": (raw - m_bg_mode) / ((m_cell_prt - m_bg_mode) or 1.0),
    }


def _parse_patch(stem: str):
    m = _PATCH_RE.match(stem)
    return (m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))) if m else None


def _resolve_raw_folders(raw_data_root: Path | None = None) -> dict[str, dict[str, Path]]:
    """Return raw CZI folder map. Precedence: explicit root > local > cluster."""
    if raw_data_root is not None:
        # Build map from an explicit root containing the four experiment folders
        # (same subfolder names as DS_RAW_FOLDERS_LOCAL)
        local = {ds: {cond: raw_data_root / p.relative_to(_LOCAL_FA)
                      for cond, p in conds.items()}
                 for ds, conds in DS_RAW_FOLDERS_LOCAL.items()}
        return local
    # Auto-select: local NAS if mounted, else cluster
    probe = next(iter(next(iter(DS_RAW_FOLDERS_LOCAL.values())).values()))
    if probe.exists():
        return DS_RAW_FOLDERS_LOCAL
    return DS_RAW_FOLDERS


def _normalize_from_raw_czis(
    records: list[dict],
    ds: str,
    conditions: list[str],
    norm: str,
    channels: list[str],
    raw_folders: dict[str, dict[str, Path]],
) -> tuple[dict | None, dict | None]:
    """Normalise patches and canvas images for ALL channels × ALL norm variants.

    Returns
    -------
    patches_dict  : {(norm_key, ch_key): (N, 32, 32) float32}  or None
    canvas_frames : {(cond, frame_idx): {(norm_key, ch_key): (H, W) float32}} or None
    """
    if ds not in raw_folders:
        print(f"  [norm] no raw folder mapping for ds={ds!r}", flush=True)
        return None, None

    rolling_ball_radius = _ROLLING_BALL.get(norm)
    scale = 5.0

    frame_map: dict[tuple, list[tuple]] = defaultdict(list)
    for i, rec in enumerate(records):
        x_c = rec["canvas_cx"] + PAD_SIZE
        y_c = rec["canvas_cy"] + PAD_SIZE
        frame_map[(rec["condition_name"], rec["frame_idx"])].append(
            (i, x_c, y_c, rec["ps"])
        )

    n      = len(records)
    all_chs = ['pax'] + [c for c in channels if c != 'pax']

    patches_dict: dict[tuple, np.ndarray] = {
        (nk, ch): np.zeros((n, 32, 32), dtype=np.float32)
        for nk in NORM_KEYS for ch in all_chs
    }
    canvas_frames: dict[tuple, dict[tuple, np.ndarray]] = {}

    for cond in conditions:
        raw_folder = raw_folders.get(ds, {}).get(cond)
        if raw_folder is None or not raw_folder.exists():
            print(f"  [norm] raw folder missing for {ds}/{cond}: {raw_folder}", flush=True)
            continue

        filenames = _pp.list_image_files(str(raw_folder), file_type="czi")
        if not filenames:
            print(f"  [norm] no CZI files in {raw_folder}", flush=True)
            continue

        cond_frame_idxs = sorted({fi for (c, fi) in frame_map if c == cond})
        print(f"  [norm] {ds}/{cond}: normalising {len(cond_frame_idxs)} frames …", flush=True)

        for fi in cond_frame_idxs:
            if fi >= len(filenames):
                print(f"    WARNING: frame_idx={fi} out of range ({len(filenames)} files)", flush=True)
                continue

            filename = filenames[fi]
            raw = _pp._load_raw_squeezed(str(raw_folder), filename, "czi")

            # Paxillin (with optional rolling ball)
            pax_raw = _pp._extract_channel(raw, _CZI_CH['pax'], filename, "czi").astype(np.float32)
            if rolling_ball_radius is not None:
                _s = 255.0 * 255.0
                pax_raw = _pp.apply_rolling_ball(pax_raw * _s, radius=rolling_ball_radius) / _s

            # Segmentation masks (always from paxillin)
            seg_input = _pp._extract_channel(raw, _SEG_PARAMS["seg_ch"], filename, "czi")
            seg = _pp.segment_cell_mask(
                seg_input,
                threshold=_SEG_PARAMS["threshold"],
                close_size=_SEG_PARAMS["close_size"],
                min_size_initial=_SEG_PARAMS["min_size_initial"],
                min_size_post_close=_SEG_PARAMS["min_size_post_close"],
                min_size_final=_SEG_PARAMS["min_size_final"],
            ).astype(bool)
            seg_all = _segment_exclude_mask(seg_input, min_size_final=1000)
            true_bg = ~seg_all

            # QC print for pax background
            pax_bg = pax_raw[true_bg] if true_bg.any() else np.array([], dtype=np.float32)
            out_mean   = float(np.mean(pax_bg))   if pax_bg.size else float('nan')
            out_median = float(np.median(pax_bg)) if pax_bg.size else float('nan')
            if pax_bg.size:
                _cnt = np.clip(np.round(pax_bg * 65536).astype(int), 0, 65535)
                out_mode = float(np.bincount(_cnt).argmax()) / 65536.0
            else:
                out_mode = float('nan')
            S = 65536
            print(f"    {filename}: bg  mean={out_mean*S:6.1f}  median={out_median*S:5.1f}"
                  f"  mode={out_mode*S:5.1f}  (×2^16)", flush=True)

            # Compute all normalizations for every channel
            frame_norms: dict[tuple, np.ndarray] = {}
            padded:      dict[tuple, np.ndarray] = {}

            for ch_key in all_chs:
                czi_idx = _CZI_CH.get(ch_key)
                if czi_idx is None or czi_idx >= raw.shape[0]:
                    continue
                ch_arr = (pax_raw if ch_key == 'pax'
                          else _pp._extract_channel(raw, czi_idx, filename, "czi").astype(np.float32))
                bg_px   = ch_arr[true_bg] if true_bg.any() else np.array([0.0], dtype=np.float32)
                cell_px = ch_arr[seg]     if seg.any()     else np.array([1.0], dtype=np.float32)

                ch_norms = _compute_channel_norms(ch_arr, bg_px, cell_px, scale, label=f"{ds}/{ch_key}/{filename}")
                for nk, norm_arr in ch_norms.items():
                    frame_norms[(nk, ch_key)] = norm_arr
                    padded[(nk, ch_key)] = _pp.image_padding(
                        norm_arr, PAD_SIZE, float(np.mean(norm_arr))
                    )

            canvas_frames[(cond, fi)] = frame_norms

            # Extract patches for all norm × ch combinations
            for (rec_i, x_c, y_c, ps) in frame_map[(cond, fi)]:
                half = ps // 2
                r0, r1 = y_c - half, y_c + half
                c0, c1 = x_c - half, x_c + half
                for key, pad_arr in padded.items():
                    if key in patches_dict:
                        patches_dict[key][rec_i] = pad_arr[r0:r1, c0:c1].astype(np.float32)

    if not canvas_frames:
        return None, None
    return patches_dict, canvas_frames


def pack_dataset(ds: str, conditions: list[str], norm: str = "cio",
                 patches_root: Path | None = None,
                 tif_patches_root: Path | None = None,
                 raw_data_root: Path | None = None) -> Path | None:
    """Pack data.h5 for one dataset.

    TIF patches (FA locations) are read from tif_patches_root (defaults to
    patches_root). This lets you re-normalise from raw CZI files using FA
    detections produced by a prior patchprep run in a different directory.
    """
    if patches_root is None:
        patches_root = DATA_ROOT / f"ae_results/patches/{norm}"
    if tif_patches_root is None:
        tif_patches_root = patches_root
    out_dir  = patches_root / ds
    out_path = out_dir / "data.h5"

    all_records:  list[dict]       = []
    all_patches:  list[np.ndarray] = []
    img_meta_rows: list[dict]      = []
    global_frame_idx = 0

    ds_ch    = DS_CHANNEL[ds]
    channels = (["pax", ds_ch, "zyx", "act"] if ds_ch not in ("pax", "zyx", "act")
                else ["pax", "zyx", "act"])
    all_chs = channels  # ['pax', ds_ch, 'zyx', 'act'] — same order as channels
    frame_arrays: dict[tuple, list] = {
        (nk, ch): [] for nk in NORM_KEYS for ch in all_chs
    }

    for cond in conditions:
        patch_dir = tif_patches_root / ds / cond / PATCH_SUBDIR
        tifs = sorted(patch_dir.glob("*.tif")) if patch_dir.exists() else []
        if not tifs:
            print(f"  SKIP {ds}/{cond}: no TIF patches in {patch_dir}", flush=True)
            continue
        print(f"  {ds}/{cond}: {len(tifs)} patches (TIF) …", flush=True)

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
        all_records.extend(records)

        patches = np.stack([tifffile.imread(str(t)).astype(np.float32) for t in tifs])
        all_patches.append(patches)
        print(f"    patches/raw: {patches.shape}", flush=True)

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

    # ── Normalise from raw CZI → patches/cio_*, images/* ────────────────────
    raw_folders = _resolve_raw_folders(raw_data_root)
    patches_dict, canvas_frames = _normalize_from_raw_czis(
        all_records, ds, conditions, norm, channels, raw_folders
    )

    if canvas_frames is not None:
        _h_ref = _w_ref = None
        for row in img_meta_rows:
            cond = row["condition_name"]
            fi   = row["frame_idx"]
            frame_norms = canvas_frames.get((cond, fi), {})
            if _h_ref is None:
                for arr in frame_norms.values():
                    _h_ref, _w_ref = arr.shape; break
            for nk in NORM_KEYS:
                for ch in all_chs:
                    arr = frame_norms.get((nk, ch))
                    if arr is not None:
                        frame_arrays[(nk, ch)].append(arr.astype(np.float32))
                    else:
                        h = _h_ref or 1024; w = _w_ref or 1024
                        frame_arrays[(nk, ch)].append(np.zeros((h, w), dtype=np.float32))
                        print(f"    WARNING: {nk}/{ch} missing for {cond}_f{fi:04d}", flush=True)
    else:
        print(f"  WARNING: canvas images not available", flush=True)

    # ── Write HDF5 ─────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(out_path), "w") as f:
        f.create_dataset("patches/raw", data=patches_all,
                         compression="gzip", compression_opts=4)
        print(f"  patches/raw saved: {patches_all.shape}", flush=True)

        if patches_dict is not None:
            for nk in NORM_KEYS:
                for ch in all_chs:
                    key = (nk, ch)
                    f.create_dataset(f"patches/{nk}/{ch}", data=patches_dict[key],
                                     compression="gzip", compression_opts=4)
            print(f"  patches/{{{','.join(NORM_KEYS)}}} × {all_chs} saved", flush=True)

        for nk in NORM_KEYS:
            for ch in all_chs:
                frames = frame_arrays.get((nk, ch))
                if frames:
                    stack = np.stack(frames)
                    f.create_dataset(f"images/{nk}/{ch}", data=stack,
                                     compression="gzip", compression_opts=4)
        # Backward-compat alias
        if frame_arrays.get(("cio_inlier", "pax")):
            stack = np.stack(frame_arrays[("cio_inlier", "pax")])
            f.create_dataset("images/raw", data=stack,
                             compression="gzip", compression_opts=4)
        print(f"  images/{{{','.join(NORM_KEYS)}}} × {all_chs} saved", flush=True)

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
    ap.add_argument("--norm", default="cio_inlier_med", choices=ALL_NORMS,
                    help="Normalisation variant (default: cio_inlier_med)")
    ap.add_argument("--patches-root", default=None, metavar="PATH",
                    help="Output dir for data.h5 files (default: DATA_ROOT/ae_results/patches/{norm})")
    ap.add_argument("--tif-patches-root", default=None, metavar="PATH",
                    help="Root containing pre-detected TIF patches in {ds}/{cond}/tiff_patches32_mr10/ "
                         "(default: same as --patches-root; use this to reuse FA detections from a "
                         "prior CIO run, e.g. /mnt/p/image_service/data/FA_patch_data/cio)")
    ap.add_argument("--raw-data-root", default=None, metavar="PATH",
                    help="Root folder containing the raw CZI experiment subfolders "
                         "(auto-detected if omitted)")
    args = ap.parse_args()

    pr  = Path(args.patches_root)     if args.patches_root     else None
    tpr = Path(args.tif_patches_root) if args.tif_patches_root else None
    rdr = Path(args.raw_data_root)    if args.raw_data_root    else None

    print(f"Packing data.h5 [{args.norm}] for {args.datasets} × {args.conditions}\n", flush=True)
    if pr:  print(f"  patches-root    : {pr}")
    if tpr: print(f"  tif-patches-root: {tpr}")
    if rdr: print(f"  raw-data-root   : {rdr}")
    for ds in args.datasets:
        print(f"=== {ds} ===", flush=True)
        pack_dataset(ds, args.conditions, norm=args.norm,
                     patches_root=pr, tif_patches_root=tpr, raw_data_root=rdr)

    print("\nAll done.", flush=True)


if __name__ == "__main__":
    main()
