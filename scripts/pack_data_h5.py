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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import subcellae.dataprep.patch_prep as _pp

_PATCH_RE = re.compile(r'^(.+)_f(\d+)x(\d+)y(\d+)ps(\d+)$')


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
) -> tuple[np.ndarray | None, np.ndarray | None, dict | None]:
    """Normalise patches and generate canvas images from raw CZI files.

    Applies cell-based normalisation (inlier-trimmed-mean and median variants)
    so that patch and canvas pixel values are on the same scale.

    Prints per-CZI outside-cell mean and median for QC.

    Returns
    -------
    patches_inlier  : (N, 32, 32) float32, or None
    patches_med     : (N, 32, 32) float32, or None
    canvas_frames   : {(cond, frame_idx): {ch_key: (H, W) float32}}
                      inlier-normalised full frame per channel, or None
    """
    if ds not in raw_folders:
        print(f"  [norm] no raw folder mapping for ds={ds!r}", flush=True)
        return None, None, None

    rolling_ball_radius = _ROLLING_BALL.get(norm)
    scale = 5.0

    # Index records by (condition, frame_idx)
    frame_map: dict[tuple, list[tuple]] = defaultdict(list)
    for i, rec in enumerate(records):
        x_c = rec["canvas_cx"] + PAD_SIZE
        y_c = rec["canvas_cy"] + PAD_SIZE
        frame_map[(rec["condition_name"], rec["frame_idx"])].append(
            (i, x_c, y_c, rec["ps"])
        )

    n = len(records)
    patches_inlier = np.zeros((n, 32, 32), dtype=np.float32)
    patches_med    = np.zeros((n, 32, 32), dtype=np.float32)
    canvas_frames: dict[tuple, dict[str, np.ndarray]] = {}

    ds_marker_ch = DS_CHANNEL.get(ds, "vinc")  # marker channel abbreviation
    # All channels to extract from CZI (pax always first for normalisation)
    all_chs = ['pax'] + [c for c in channels if c != 'pax']

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

            # ── paxillin: rolling ball → segment → normalise ──────────────────
            pax_raw = _pp._extract_channel(raw, _CZI_CH['pax'], filename, "czi")
            if rolling_ball_radius is not None:
                _s = 255.0 * 255.0
                pax_raw = _pp.apply_rolling_ball(pax_raw * _s, radius=rolling_ball_radius) / _s

            seg_input = _pp._extract_channel(raw, _SEG_PARAMS["seg_ch"], filename, "czi")
            seg = _pp.segment_cell_mask(
                seg_input,
                threshold=_SEG_PARAMS["threshold"],
                close_size=_SEG_PARAMS["close_size"],
                min_size_initial=_SEG_PARAMS["min_size_initial"],
                min_size_post_close=_SEG_PARAMS["min_size_post_close"],
                min_size_final=_SEG_PARAMS["min_size_final"],
            ).astype(float)

            outside_mask = (1 - seg).astype(bool)
            outside_vals = pax_raw[outside_mask]
            out_mean   = float(np.mean(outside_vals))   if outside_vals.size else float('nan')
            out_median = float(np.median(outside_vals)) if outside_vals.size else float('nan')
            print(f"    {filename}: outside-cell pax  mean={out_mean:.5f}  median={out_median:.5f}",
                  flush=True)

            pax_inlier = _pp.normalize_cell_insideoutside_inlier(pax_raw, seg, scale=scale)
            pax_med    = _pp.normalize_cell_insideoutside_med(pax_raw,    seg, scale=scale)

            pax_inlier_pad = _pp.image_padding(pax_inlier, PAD_SIZE, float(np.mean(pax_inlier)))
            pax_med_pad    = _pp.image_padding(pax_med,    PAD_SIZE, float(np.mean(pax_med)))

            # ── other channels: normalise with same seg mask (inlier variant) ─
            ch_canvas: dict[str, np.ndarray] = {'pax': pax_inlier}
            for ch_key in all_chs:
                if ch_key == 'pax':
                    continue
                czi_idx = _CZI_CH.get(ch_key)
                if czi_idx is None or czi_idx >= raw.shape[0]:
                    continue
                ch_raw = _pp._extract_channel(raw, czi_idx, filename, "czi").astype(np.float32)
                ch_norm = _pp.normalize_cell_insideoutside_inlier(ch_raw, seg, scale=scale)
                ch_canvas[ch_key] = ch_norm

            canvas_frames[(cond, fi)] = ch_canvas

            # ── extract patches from padded frames ────────────────────────────
            for (rec_i, x_c, y_c, ps) in frame_map[(cond, fi)]:
                half = ps // 2
                r0, r1 = y_c - half, y_c + half
                c0, c1 = x_c - half, x_c + half
                patches_inlier[rec_i] = pax_inlier_pad[r0:r1, c0:c1].astype(np.float32)
                patches_med[rec_i]    = pax_med_pad[r0:r1, c0:c1].astype(np.float32)

    if not canvas_frames:
        return None, None, None

    return patches_inlier, patches_med, canvas_frames


def pack_dataset(ds: str, conditions: list[str], norm: str = "cio",
                 patches_root: Path | None = None,
                 raw_data_root: Path | None = None) -> Path | None:
    if patches_root is None:
        patches_root = DATA_ROOT / f"ae_results/patches/{norm}"
    out_dir  = patches_root / ds
    out_path = out_dir / "data.h5"

    all_records:  list[dict]       = []
    all_patches:  list[np.ndarray] = []
    img_meta_rows: list[dict]      = []
    # frame_arrays: {ch_key: [(H,W) float32, ...]}  — one entry per canvas frame
    frame_arrays: dict[str, list[np.ndarray]] = {}
    global_frame_idx = 0

    ds_ch    = DS_CHANNEL[ds]
    channels = (["pax", ds_ch, "zyx", "act"] if ds_ch not in ("pax", "zyx", "act")
                else ["pax", "zyx", "act"])
    for ch in channels:
        frame_arrays[ch] = []

    for cond in conditions:
        patch_dir = patches_root / ds / cond / PATCH_SUBDIR

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

        all_records.extend(records)

        patches = np.stack([tifffile.imread(str(t)).astype(np.float32) for t in tifs])
        all_patches.append(patches)
        print(f"    patches/raw: {patches.shape}", flush=True)

        # image meta rows (frame indices updated after canvas is known)
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

    # ── Normalise from raw CZI → patches/cio_inlier, patches/cio_med, images/* ─
    raw_folders = _resolve_raw_folders(raw_data_root)
    patches_inlier, patches_med, canvas_frames = _normalize_from_raw_czis(
        all_records, ds, conditions, norm, channels, raw_folders
    )

    # Build frame_arrays in the same order as img_meta_rows
    if canvas_frames is not None:
        for row in img_meta_rows:
            cond = row["condition_name"]
            fi   = row["frame_idx"]
            ch_canvas = canvas_frames.get((cond, fi), {})
            for ch in channels:
                arr = ch_canvas.get(ch)
                if arr is not None:
                    frame_arrays[ch].append(arr.astype(np.float32))
                else:
                    # fall back to zeros if this channel wasn't in the CZI
                    h, w = next((a.shape for a in
                                 sum(frame_arrays.values(), []) if a is not None), (1024, 1024))
                    frame_arrays[ch].append(np.zeros((h, w), dtype=np.float32))
                    print(f"    WARNING: {ch} missing for {cond}_f{fi:04d}, using zeros", flush=True)
    else:
        print(f"  WARNING: canvas images not available — images/* will be empty", flush=True)

    # ── Write HDF5 ─────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(out_path), "w") as f:
        f.create_dataset("patches/raw", data=patches_all,
                         compression="gzip", compression_opts=4)
        print(f"  patches/raw saved: {patches_all.shape}", flush=True)

        if patches_inlier is not None:
            f.create_dataset("patches/cio_inlier", data=patches_inlier,
                             compression="gzip", compression_opts=4)
            f.create_dataset("patches/cio_med", data=patches_med,
                             compression="gzip", compression_opts=4)
            print(f"  patches/cio_inlier and patches/cio_med saved", flush=True)

        if frame_arrays.get("pax"):
            stack = np.stack(frame_arrays["pax"])
            f.create_dataset("images/raw", data=stack,
                             compression="gzip", compression_opts=4)
            print(f"  images/raw (pax, inlier-normed): {stack.shape}", flush=True)

        for ch in channels:
            if ch != "pax" and frame_arrays.get(ch):
                stack = np.stack(frame_arrays[ch])
                f.create_dataset(f"images/{ch}", data=stack,
                                 compression="gzip", compression_opts=4)
                print(f"  images/{ch}: {stack.shape}", flush=True)

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
                    help="Override patches root dir (default: DATA_ROOT/ae_results/patches/{norm})")
    ap.add_argument("--raw-data-root", default=None, metavar="PATH",
                    help="Root folder containing the raw CZI experiment subfolders "
                         "(auto-detected if omitted)")
    args = ap.parse_args()

    pr = Path(args.patches_root)  if args.patches_root  else None
    rdr = Path(args.raw_data_root) if args.raw_data_root else None

    print(f"Packing data.h5 [{args.norm}] for {args.datasets} × {args.conditions}\n", flush=True)
    if pr:  print(f"  patches-root : {pr}")
    if rdr: print(f"  raw-data-root: {rdr}")
    for ds in args.datasets:
        print(f"=== {ds} ===", flush=True)
        pack_dataset(ds, args.conditions, norm=args.norm,
                     patches_root=pr, raw_data_root=rdr)

    print("\nAll done.", flush=True)


if __name__ == "__main__":
    main()
