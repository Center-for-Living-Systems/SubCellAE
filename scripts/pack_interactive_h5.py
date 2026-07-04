"""
Pack a (variant, dataset) result directory into a structured HDF5
suitable for the interactive patch viewer (view_interactive.py).

Reads:
  {result_dir}/latents_newdata.csv
  {result_dir}/analysis/analysis_results.csv   (UMAP_1, UMAP_2)
  {result_dir}/fa_cls_lat8/predictions_all.csv
  {result_dir}/pos_cls_lat8/predictions_all.csv
  {result_dir}/recon/patches_raw.tif  + patches_index.csv
  {result_dir}/recon/patches_recon.tif
  {result_dir}/recon/images_raw.tif   + images_index.csv

Writes:
  {result_dir}/interactive.h5   (or --out path)

Usage:
    python scripts/pack_interactive_h5.py <result_dir>
    python scripts/pack_interactive_h5.py <result_dir> --out /tmp/my.h5
    python scripts/pack_interactive_h5.py <result_dir> --image-scale 0.5
"""

from __future__ import annotations

import argparse
import base64
import io
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import tifffile
from PIL import Image

# Regex matching the patch filename pattern:
#   {group}_f{img_id}x{x_c}y{y_c}ps{ps}[.tif]
_COORD_RE = re.compile(r'^(.+_f\d+)x(\d+)y(\d+)ps(\d+)')
_PATCH_COND_RE = re.compile(r'^(.+?)_f(\d+)x(\d+)y(\d+)ps(\d+)')


def _parse_patch_coords(filename: str):
    """Return (group, x_c, y_c, ps) parsed from the patch filename stem.

    Coordinates (x_c, y_c) are in the *padded* image space; subtract
    pad_size from each to get the canvas coordinates used by the viewer.
    Returns (None, None, None, None) on mismatch.
    """
    stem = Path(str(filename)).stem
    m = _COORD_RE.match(stem)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
    return None, None, None, None


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def _encode_patch_b64(arr_f32: np.ndarray, zoom: int = 4) -> str:
    """Encode a (H, W) float32 [0,1] array as a base64 PNG string.

    The patch is zoomed by `zoom` using nearest-neighbour so it renders
    clearly in the Bokeh hover tooltip at ~128 px.
    """
    img = _to_uint8(arr_f32)
    if zoom > 1:
        img = img.repeat(zoom, axis=0).repeat(zoom, axis=1)
    buf = io.BytesIO()
    Image.fromarray(img, mode='L').save(buf, format='PNG', optimize=True)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def _scale_allch_image(arr: np.ndarray, scale: float) -> np.ndarray:
    """Downscale a (C, H, W) float32 [0,1] image by scale factor."""
    if scale == 1.0:
        return arr
    C, H, W = arr.shape
    nH, nW = max(1, int(H * scale)), max(1, int(W * scale))
    result = np.empty((C, nH, nW), dtype=np.float32)
    for ci in range(C):
        pil = Image.fromarray(_to_uint8(arr[ci]), mode='L')
        result[ci] = np.array(pil.resize((nW, nH), Image.LANCZOS), dtype=np.float32) / 255.0
    return result


def _get_czi_channel_names(path: Path) -> list | None:
    """Return list of channel display names from CZI XML metadata, or None."""
    try:
        import czifile as _czifile
        import xml.etree.ElementTree as ET
        with _czifile.CziFile(str(path)) as czi:
            xml_str = czi.metadata()
        root = ET.fromstring(xml_str)
        # Prefer DisplaySetting — has user-friendly short names
        channels = root.findall('.//DisplaySetting/Channels/Channel')
        if channels:
            names = [ch.get('ShortName') or ch.get('Name') or '' for ch in channels]
            names = [n.strip() for n in names if n.strip()]
            if names:
                return names
        # Fallback: Dimensions/Channels
        channels = root.findall('.//Dimensions/Channels/Channel')
        if channels:
            names = [ch.findtext('Name') or ch.get('Name') or '' for ch in channels]
            names = [n.strip() for n in names if n.strip()]
            if names:
                return names
        return None
    except Exception as e:
        print(f"[pack]   WARN: could not extract channel names from {path.name}: {e}",
              file=sys.stderr)
        return None


def _extract_allch_patches(
    df: pd.DataFrame,
    image_folder_map: dict,   # {condition_str | None: Path}
    pad: int,
) -> tuple[np.ndarray | None, dict | None, list | None]:
    """Extract all CZI channels for every patch; also return full-canvas images.

    Returns
    -------
    patches       : (N, C, ps, ps) float32 array, or None on failure
    images        : dict {group_name: (C, H, W) float32}, or None on failure
    channel_names : list of str channel names extracted from CZI metadata, or None

    Normalisation is per-channel over the full loaded image (1%–99% percentile
    stretch), so brightness is consistent across patches from the same image.

    image_folder_map keys:
        str  → applies only to patches whose filename prefix matches that key
        None → fallback / applies to all conditions
    Returns (None, None, None) if no patches could be loaded.
    """
    try:
        import czifile as _czifile
    except ImportError:
        print("[pack]   WARN: czifile not installed – skipping multi-channel packing.", file=sys.stderr)
        return None, None, None

    # ── helpers ──────────────────────────────────────────────────────────────
    def _get_folder(cond: str) -> Path | None:
        return image_folder_map.get(cond, image_folder_map.get(None))

    def _load_and_norm_czi(path: Path) -> np.ndarray:
        """Load CZI and percentile-normalise each channel over the full image."""
        raw = _czifile.imread(str(path)).squeeze().astype(np.float32) / (255.0 * 255.0)
        if raw.ndim == 2:
            raw = raw[np.newaxis]          # (1, H, W)
        out = np.empty_like(raw)
        for ch in range(raw.shape[0]):
            arr = raw[ch]
            p1  = float(np.percentile(arr, 1))
            p99 = float(np.percentile(arr, 99))
            if p99 > p1:
                out[ch] = np.clip((arr - p1) / (p99 - p1), 0.0, 1.0)
            else:
                out[ch] = np.zeros_like(arr)
        return out   # (C, H, W) float32 in [0, 1]

    # ── build (condition, frame_id) → sorted CZI list cache ──────────────────
    czi_list_cache: dict = {}   # condition → sorted list of Path

    def _czi_files_for_cond(cond: str):
        if cond not in czi_list_cache:
            folder = _get_folder(cond)
            if folder is None or not folder.is_dir():
                czi_list_cache[cond] = []
            else:
                czi_list_cache[cond] = sorted(folder.glob("*.czi"))
        return czi_list_cache[cond]

    # ── main loop ─────────────────────────────────────────────────────────────
    czi_arr_cache: dict = {}    # (cond, frame_id) → normalised (C, H, W) array
    key_to_group:  dict = {}    # (cond, frame_id) → group_name string

    n_patches = len(df)
    allch_rows: list = []

    for idx, row in df.iterrows():
        fname = Path(str(row.get('filename', ''))).stem
        m = _PATCH_COND_RE.match(fname)
        if m is None:
            allch_rows.append(None)
            continue

        cond      = m.group(1)
        frame_id  = int(m.group(2))
        ps        = int(m.group(5))
        canvas_cx = int(float(row.get('canvas_cx', m.group(3))))
        canvas_cy = int(float(row.get('canvas_cy', m.group(4))))

        cache_key = (cond, frame_id)
        key_to_group.setdefault(cache_key, f'{cond}_f{frame_id}')
        if cache_key not in czi_arr_cache:
            czi_files = _czi_files_for_cond(cond)
            if frame_id >= len(czi_files):
                czi_arr_cache[cache_key] = None
                print(f"[pack]   WARN: frame {frame_id} out of range for condition {cond!r} "
                      f"({len(czi_files)} files found)", file=sys.stderr)
            else:
                try:
                    czi_arr_cache[cache_key] = _load_and_norm_czi(czi_files[frame_id])
                except Exception as e:
                    czi_arr_cache[cache_key] = None
                    print(f"[pack]   WARN: could not load {czi_files[frame_id]}: {e}", file=sys.stderr)

        czi_norm = czi_arr_cache[cache_key]
        if czi_norm is None:
            allch_rows.append(None)
            continue

        half = ps // 2
        ys, ye = canvas_cy - half, canvas_cy + half
        xs, xe = canvas_cx - half, canvas_cx + half
        # Clamp to image bounds
        C, H, W = czi_norm.shape
        ys, ye = max(0, ys), min(H, ye)
        xs, xe = max(0, xs), min(W, xe)
        patch = czi_norm[:, ys:ye, xs:xe]   # (C, ps, ps) – may be smaller at edges

        if patch.shape[1] != ps or patch.shape[2] != ps:
            # Pad to full ps×ps if near image edge
            p = np.zeros((C, ps, ps), dtype=np.float32)
            p[:, :patch.shape[1], :patch.shape[2]] = patch
            patch = p

        allch_rows.append(patch.astype(np.float32))

    # Filter None rows and check we have at least something
    valid = [r for r in allch_rows if r is not None]
    if not valid:
        print("[pack]   WARN: no patches could be loaded from CZI files.", file=sys.stderr)
        return None, None, None

    n_ch = valid[0].shape[0]
    ps0  = valid[0].shape[1]
    out  = np.zeros((n_patches, n_ch, ps0, ps0), dtype=np.float32)
    for i, patch in enumerate(allch_rows):
        if patch is not None:
            out[i] = patch

    print(f"[pack]   Multi-channel patches: {n_patches} × {n_ch} channels × {ps0}×{ps0}px")

    # Collect full-canvas images from cache
    images_allch: dict = {}
    for key in sorted(czi_arr_cache):
        arr = czi_arr_cache[key]
        if arr is not None:
            group_name = key_to_group.get(key, f'{key[0]}_f{key[1]}')
            images_allch[group_name] = arr
    print(f"[pack]   Full-canvas allch images: {len(images_allch)} groups")

    # Extract channel names from the first successfully loaded CZI
    channel_names = None
    for key in sorted(czi_arr_cache):
        if czi_arr_cache[key] is not None:
            cond, frame_id = key
            czi_files = _czi_files_for_cond(cond)
            if frame_id < len(czi_files):
                channel_names = _get_czi_channel_names(czi_files[frame_id])
                if channel_names:
                    print(f"[pack]   Channel names: {channel_names}")
                    break

    return out, (images_allch if images_allch else None), channel_names


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('result_dir',
                    help='Result directory at the variant/dataset level '
                         '(e.g. ae_results/other_paxillin/baseline/vinc)')
    ap.add_argument('--out', default=None,
                    help='Output HDF5 path (default: result_dir/interactive.h5)')
    ap.add_argument('--pad-size', type=int, default=64,
                    help='Pad size used during patch extraction (default: 64)')
    ap.add_argument('--image-scale', type=float, default=1.0,
                    help='Downscale full canvas images by this factor to save space '
                         '(default: 1.0 = no downscale; try 0.5 for large datasets)')
    ap.add_argument(
        '--image-folder', dest='image_folders', action='append', default=[],
        metavar='[CONDITION:]FOLDER',
        help=(
            'CZI image folder for multi-channel packing. '
            'Format: "FOLDER" (all conditions) or "CONDITION:FOLDER". '
            'Repeat for multiple conditions. '
            'Requires czifile. Example: '
            '--image-folder control:/data/Control --image-folder ycomp:/data/Ycomp'
        ),
    )
    args = ap.parse_args()

    result_dir = Path(args.result_dir)
    out_h5 = Path(args.out) if args.out else result_dir / 'interactive.h5'
    pad = args.pad_size

    if not result_dir.is_dir():
        print(f"[pack] ERROR: {result_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    print(f"[pack] {result_dir}")
    print(f"[pack] → {out_h5}")

    # ── Load and merge metadata CSVs ─────────────────────────────────────────
    latents_csv  = result_dir / 'latents_newdata.csv'
    fa_pred_csv  = result_dir / 'fa_cls_lat8'  / 'predictions_all.csv'
    pos_pred_csv = result_dir / 'pos_cls_lat8' / 'predictions_all.csv'
    analysis_csv = result_dir / 'analysis'     / 'analysis_results.csv'

    if not latents_csv.exists():
        print(f"[pack] ERROR: {latents_csv} not found — run ae_apply first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(latents_csv)
    print(f"[pack]   {len(df)} patches in latents_newdata.csv")

    # Parse patch filename → group, canvas_cx, canvas_cy, ps
    parsed = df['filename'].apply(
        lambda f: pd.Series(
            dict(zip(['patch_group', 'x_c', 'y_c', 'ps'],
                     _parse_patch_coords(f)))
        )
    )
    df = pd.concat([df, parsed], axis=1)
    df['canvas_cx'] = pd.to_numeric(df['x_c'], errors='coerce') - pad
    df['canvas_cy'] = pd.to_numeric(df['y_c'], errors='coerce') - pad

    # FA type predictions
    if fa_pred_csv.exists():
        fa = pd.read_csv(fa_pred_csv)
        proba_cols = [c for c in fa.columns if c.startswith('proba_')]
        fa_sub = fa[['filename', 'pred_label'] + proba_cols].copy()
        fa_sub = fa_sub.rename(columns={'pred_label': 'fa_pred'})
        fa_sub = fa_sub.rename(columns={c: f'fa_{c}' for c in proba_cols})
        df = df.merge(fa_sub, on='filename', how='left')
        print(f"[pack]   FA predictions merged ({fa_sub['fa_pred'].nunique()} classes)")
    else:
        print(f"[pack]   WARN: {fa_pred_csv} not found")

    # Position predictions
    if pos_pred_csv.exists():
        pos = pd.read_csv(pos_pred_csv)
        proba_cols = [c for c in pos.columns if c.startswith('proba_')]
        pos_sub = pos[['filename', 'pred_label'] + proba_cols].copy()
        pos_sub = pos_sub.rename(columns={'pred_label': 'pos_pred'})
        pos_sub = pos_sub.rename(columns={c: f'pos_{c}' for c in proba_cols})
        df = df.merge(pos_sub, on='filename', how='left')
        print(f"[pack]   Position predictions merged ({pos_sub['pos_pred'].nunique()} classes)")
    else:
        print(f"[pack]   WARN: {pos_pred_csv} not found")

    # UMAP coordinates from analysis
    if analysis_csv.exists():
        ana = pd.read_csv(analysis_csv)
        umap_cols = [c for c in ana.columns if c.upper().startswith('UMAP')]
        if len(umap_cols) >= 2:
            rename_map = {umap_cols[0]: 'UMAP_1', umap_cols[1]: 'UMAP_2'}
            ana_sub = ana[['filename'] + umap_cols[:2]].rename(columns=rename_map)
            df = df.merge(ana_sub, on='filename', how='left')
            print(f"[pack]   UMAP coordinates merged")
        else:
            print(f"[pack]   WARN: no UMAP columns found in {analysis_csv}")
    else:
        print(f"[pack]   WARN: {analysis_csv} not found — viewer will fall back to latent dims")

    # ── Load patch stacks ─────────────────────────────────────────────────────
    recon_dir         = result_dir / 'recon'
    patches_raw_tif   = recon_dir  / 'patches_raw.tif'
    patches_recon_tif = recon_dir  / 'patches_recon.tif'
    patches_idx_csv   = recon_dir  / 'patches_index.csv'

    patches_raw_arr = patches_recon_arr = None

    if patches_raw_tif.exists() and patches_idx_csv.exists():
        patch_idx   = pd.read_csv(patches_idx_csv)
        raw_stack   = tifffile.imread(str(patches_raw_tif))
        recon_stack = tifffile.imread(str(patches_recon_tif))
        n_frames, H, W = raw_stack.shape[:3]

        name_to_frame = {row['name']: int(row['frame'])
                         for _, row in patch_idx.iterrows()}
        stems         = df['filename'].apply(lambda f: Path(str(f)).stem)
        frame_indices = stems.map(name_to_frame)

        n = len(df)
        patches_raw_arr   = np.zeros((n, H, W), dtype=np.float32)
        patches_recon_arr = np.zeros((n, H, W), dtype=np.float32)
        for i, frame in enumerate(frame_indices):
            if pd.notna(frame):
                f = int(frame)
                patches_raw_arr[i]   = raw_stack[f]
                patches_recon_arr[i] = recon_stack[f]

        print(f"[pack]   {n} patches loaded from TIFFs ({H}×{W}px)")

        # Encode patches as base64 PNG for Bokeh hover tooltips
        print(f"[pack]   Encoding patches as base64 PNG for hover tooltips …")
        df['raw_b64']   = [_encode_patch_b64(patches_raw_arr[i])   for i in range(n)]
        df['recon_b64'] = [_encode_patch_b64(patches_recon_arr[i]) for i in range(n)]
        print(f"[pack]   Done encoding")
    else:
        # Fallback: load individual raw_*/recon_* files from recon/patches/
        patches_dir = recon_dir / 'patches'
        stems = df['filename'].apply(lambda f: Path(str(f)).stem)
        first_raw = patches_dir / f'raw_{stems.iloc[0]}.tif' if patches_dir.exists() else None
        if first_raw is not None and first_raw.exists():
            sample = tifffile.imread(str(first_raw)).squeeze()
            H, W = sample.shape[:2]
            n = len(df)
            patches_raw_arr   = np.zeros((n, H, W), dtype=np.float32)
            patches_recon_arr = np.zeros((n, H, W), dtype=np.float32)
            for i, stem in enumerate(stems):
                rp = patches_dir / f'raw_{stem}.tif'
                rc = patches_dir / f'recon_{stem}.tif'
                if rp.exists():
                    patches_raw_arr[i]   = tifffile.imread(str(rp)).squeeze()
                if rc.exists():
                    patches_recon_arr[i] = tifffile.imread(str(rc)).squeeze()
            print(f"[pack]   {n} patches loaded from recon/patches/ ({H}×{W}px)")
            print(f"[pack]   Encoding patches as base64 PNG for hover tooltips …")
            df['raw_b64']   = [_encode_patch_b64(patches_raw_arr[i])   for i in range(n)]
            df['recon_b64'] = [_encode_patch_b64(patches_recon_arr[i]) for i in range(n)]
            print(f"[pack]   Done encoding")
        else:
            print(f"[pack]   WARN: patch TIFFs not found in {recon_dir}")

    # ── Load full canvas images ───────────────────────────────────────────────
    images_raw_tif = recon_dir / 'images_raw.tif'
    images_idx_csv = recon_dir / 'images_index.csv'

    images_raw_arr = None
    img_meta_df    = None

    def _scale_image_arr(arr_stack: np.ndarray, scale: float) -> np.ndarray:
        """Downscale a (M, H, W) image stack by scale factor."""
        if scale == 1.0:
            return arr_stack
        scaled = []
        for img in arr_stack:
            pil = (Image.fromarray(_to_uint8(img), mode='L')
                   if img.ndim == 2 else Image.fromarray(_to_uint8(img)))
            new_sz = (max(1, int(pil.width * scale)), max(1, int(pil.height * scale)))
            scaled.append(np.array(pil.resize(new_sz, Image.LANCZOS), dtype=np.uint8))
        return np.stack(scaled)

    if images_raw_tif.exists() and images_idx_csv.exists():
        img_meta_df    = pd.read_csv(images_idx_csv)
        images_raw_arr = tifffile.imread(str(images_raw_tif))  # (M, H', W')
        images_raw_arr = _scale_image_arr(images_raw_arr, args.image_scale)
        print(f"[pack]   {images_raw_arr.shape[0]} canvas images loaded "
              f"({images_raw_arr.shape[-2]}×{images_raw_arr.shape[-1]}px)")
    else:
        # Fallback: load individual raw_*.tif files from recon/images/
        images_dir = recon_dir / 'images'
        raw_img_files = sorted(images_dir.glob('raw_*.tif')) if images_dir.exists() else []
        if raw_img_files:
            img_list, meta_rows = [], []
            for i, p in enumerate(raw_img_files):
                arr = tifffile.imread(str(p)).astype(np.float32)
                if arr.ndim == 3:
                    arr = arr[0]
                mx = arr.max()
                img_list.append(arr / mx if mx > 0 else arr)
                meta_rows.append({'frame': i, 'group': p.stem[4:]})  # strip 'raw_'
            images_raw_arr = _scale_image_arr(np.stack(img_list), args.image_scale)
            img_meta_df    = pd.DataFrame(meta_rows)
            print(f"[pack]   {len(img_list)} canvas images loaded from recon/images/ "
                  f"({images_raw_arr.shape[-2]}×{images_raw_arr.shape[-1]}px)")
        else:
            print(f"[pack]   WARN: image TIFFs not found in {recon_dir}")

    # ── Extract multi-channel patches from CZI ────────────────────────────────
    allch_arr    = None
    allch_images = None
    if args.image_folders:
        image_folder_map = {}
        for spec in args.image_folders:
            if ':' in spec:
                cond, folder = spec.split(':', 1)
                image_folder_map[cond] = Path(folder)
            else:
                image_folder_map[None] = Path(spec)
        print(f"[pack]   Multi-channel image folders: {image_folder_map}")
        allch_arr, allch_images, channel_names_list = _extract_allch_patches(df, image_folder_map, pad)
        if allch_images and args.image_scale != 1.0:
            allch_images = {gname: _scale_allch_image(arr, args.image_scale)
                            for gname, arr in allch_images.items()}

    # ── Write HDF5 ────────────────────────────────────────────────────────────
    print(f"[pack]   Writing HDF5 …")
    with h5py.File(out_h5, 'w') as f:
        # Metadata as UTF-8 CSV string
        f.create_dataset('meta/csv', data=df.to_csv(index=False).encode('utf-8'))

        f.attrs['pad_size']    = pad
        f.attrs['image_scale'] = args.image_scale
        f.attrs['result_dir']  = str(result_dir)
        f.attrs['n_patches']   = len(df)

        if patches_raw_arr is not None:
            f.create_dataset('patches/raw',   data=patches_raw_arr,
                             compression='gzip', compression_opts=4)
            f.create_dataset('patches/recon', data=patches_recon_arr,
                             compression='gzip', compression_opts=4)

        if allch_arr is not None:
            f.create_dataset('patches/allch', data=allch_arr,
                             compression='gzip', compression_opts=4)
            f.attrs['n_channels'] = int(allch_arr.shape[1])
            if channel_names_list:
                import json as _json
                f.attrs['channel_names'] = _json.dumps(channel_names_list)

        if allch_images:
            g = f.create_group('images/allch')
            for gname, arr in allch_images.items():
                g.create_dataset(gname, data=arr, compression='gzip', compression_opts=4)
            _n_allch = next(iter(allch_images.values())).shape[0]
            print(f"[pack]   Wrote images/allch: {len(allch_images)} groups × {_n_allch} channels")

        if images_raw_arr is not None:
            f.create_dataset('images/raw',  data=images_raw_arr,
                             compression='gzip', compression_opts=4)
            f.create_dataset('images/meta',
                             data=img_meta_df.to_csv(index=False).encode('utf-8'))

        # ── Analysis plots (MSE distribution, MSE by condition) ───────────────
        analysis_dir = result_dir / 'analysis'
        plot_names = ['mse_distribution', 'mse_by_condition_split']
        for pname in plot_names:
            p = analysis_dir / f'{pname}.png'
            if p.exists():
                f.create_dataset(f'plots/{pname}',
                                 data=np.frombuffer(p.read_bytes(), dtype=np.uint8))
                print(f"[pack]   Packed plot: {pname}.png")
            else:
                print(f"[pack]   WARN: {p} not found")

    size_mb = out_h5.stat().st_size / 1e6
    print(f"[pack]   Done — {out_h5.name}  ({size_mb:.1f} MB,  {len(df)} patches)")


if __name__ == '__main__':
    main()
