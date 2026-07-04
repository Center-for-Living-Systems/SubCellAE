"""
pack_labeler_from_prep.py
=========================
Pack a patchprep plot-output directory into an interactive.h5 for
label_patches.py, WITHOUT requiring AE latent features / predictions.

Reads:
  {plot_dir}/data_prep_record_*.csv     — patch coordinates & filenames
  patch TIF paths stored inside the CSV  — individual patch images
  CZI image folders (--image-folder)    — multi-channel full images

Writes:
  {plot_dir}/interactive.h5  (or --out path)

Usage
-----
    # All CZI files in one folder:
    python scripts/pack_labeler_from_prep.py /path/to/plot_dir \\
        --image-folder /path/to/czis \\
        --pad-size 64 --image-scale 0.5

    # Condition-specific CZI folders:
    python scripts/pack_labeler_from_prep.py /path/to/plot_dir \\
        --image-folder control:/path/to/ctrl_czis \\
        --image-folder ycomp:/path/to/ycomp_czis \\
        --pad-size 64 --image-scale 0.5

Notes
-----
- Point --plot-dir at the channel you want as the main canvas (usually ch1 /
  paxillin).  All four channels will appear in the viewer via --image-folder.
- mask_ratio >= 0.1 filtering is already baked into the patchprep output;
  no further filtering is applied here.
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

# ── reuse helpers from the existing packer ──────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from pack_interactive_h5 import (
    _COORD_RE,
    _encode_patch_b64,
    _extract_allch_patches,
    _get_czi_channel_names,
    _scale_allch_image,
    _to_uint8,
)


# ── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        'plot_dir',
        help='Directory containing data_prep_record_*.csv files '
             '(the movie_plot_dir from patchprep config).',
    )
    ap.add_argument('--out', default=None,
                    help='Output H5 path (default: plot_dir/interactive.h5)')
    ap.add_argument('--pad-size', type=int, default=64,
                    help='Pad size used during patch extraction (default: 64)')
    ap.add_argument('--image-scale', type=float, default=1.0,
                    help='Downscale full-canvas images (default: 1.0; try 0.5)')
    ap.add_argument('--major-ch', type=int, default=1,
                    help='CZI channel index shown as main canvas (default: 1)')
    ap.add_argument(
        '--image-folder', dest='image_folders', action='append', default=[],
        metavar='[CONDITION:]FOLDER',
        help='CZI folder for multi-channel packing. '
             'Repeat for multiple conditions: --image-folder ctrl:/data/ctrl',
    )
    args = ap.parse_args()

    plot_dir = Path(args.plot_dir)
    out_h5   = Path(args.out) if args.out else plot_dir / 'interactive.h5'
    pad      = args.pad_size

    if not plot_dir.is_dir():
        print(f"[pack] ERROR: {plot_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # ── 1. Read all patchprep CSVs ───────────────────────────────────────────
    csv_files = sorted(plot_dir.glob('data_prep_record_*.csv'))
    if not csv_files:
        print(f"[pack] ERROR: no data_prep_record_*.csv found in {plot_dir}",
              file=sys.stderr)
        sys.exit(1)

    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    print(f"[pack] {len(df)} patches from {len(csv_files)} CSV file(s)")

    # ── 2. Build columns expected by the labeler and _extract_allch_patches ──
    # 'filename' = patch basename (used by _extract_allch_patches regex)
    df['filename'] = df['crop_img_filename'].apply(lambda p: Path(str(p)).name)

    # canvas coordinates: patchprep stores in padded space → subtract pad
    df['canvas_cx'] = pd.to_numeric(df['x_c'], errors='coerce') - pad
    df['canvas_cy'] = pd.to_numeric(df['y_c'], errors='coerce') - pad

    # patch_group and ps from filename stem via _COORD_RE
    def _parse_fname(fname: str) -> pd.Series:
        stem = Path(str(fname)).stem
        m = _COORD_RE.match(stem)
        return pd.Series({
            'patch_group': m.group(1) if m else None,
            'ps':          int(m.group(4)) if m else 32,
        })

    parsed = df['filename'].apply(_parse_fname)
    df = pd.concat([df, parsed], axis=1)

    # condition_name: use movie_partitioned_data_dir basename as proxy
    if 'movie_partitioned_data_dir' in df.columns:
        df['condition_name'] = df['movie_partitioned_data_dir'].apply(
            lambda p: Path(str(p)).name
        )
    else:
        df['condition_name'] = 'unknown'

    print(f"[pack] patch_groups: {sorted(df['patch_group'].dropna().unique())[:6]}")

    # ── 3. Load patch TIFs for b64 hover thumbnails (optional) ───────────────
    patches_raw_arr = None

    patch_dir_col = 'movie_partitioned_data_dir'
    if patch_dir_col in df.columns:
        first_dir  = Path(str(df[patch_dir_col].iloc[0]))
        first_file = first_dir / df['filename'].iloc[0]
        if first_file.exists():
            sample = tifffile.imread(str(first_file)).squeeze()
            H_p, W_p = sample.shape[:2]
            n = len(df)
            patches_raw_arr = np.zeros((n, H_p, W_p), dtype=np.float32)
            missing = 0
            for i, row in df.iterrows():
                p = Path(str(row[patch_dir_col])) / row['filename']
                if p.exists():
                    arr = tifffile.imread(str(p)).squeeze().astype(np.float32)
                    mx  = arr.max()
                    patches_raw_arr[i] = arr / mx if mx > 0 else arr
                else:
                    missing += 1
            if missing:
                print(f"[pack] WARN: {missing} patch TIFs not found")
            print(f"[pack] {n} patch TIFs loaded ({H_p}×{W_p}px)")
            print(f"[pack] Encoding patches as base64 PNG …")
            df['raw_b64'] = [_encode_patch_b64(patches_raw_arr[i]) for i in range(n)]
        else:
            print(f"[pack] WARN: patch TIFs not found — skipping hover thumbnails")

    # ── 4. Extract multi-channel patches + full-canvas images from CZI ───────
    allch_arr        = None
    allch_images     = None
    channel_names_list = None

    if args.image_folders:
        image_folder_map: dict = {}
        for spec in args.image_folders:
            if ':' in spec:
                cond, folder = spec.split(':', 1)
                image_folder_map[cond] = Path(folder)
            else:
                image_folder_map[None] = Path(spec)
        print(f"[pack] Multi-channel image folders: {image_folder_map}")
        allch_arr, allch_images, channel_names_list = _extract_allch_patches(
            df, image_folder_map, pad
        )
        if allch_images and args.image_scale != 1.0:
            allch_images = {gname: _scale_allch_image(arr, args.image_scale)
                            for gname, arr in allch_images.items()}
    else:
        print("[pack] WARN: no --image-folder given — multi-channel data skipped")

    # ── 5. Build images/raw from allch major channel (main canvas) ────────────
    images_raw_arr = None
    img_meta_df    = None

    if allch_images:
        major_ch = args.major_ch
        img_list, meta_rows = [], []
        for i, (gname, arr) in enumerate(sorted(allch_images.items())):
            ch_img = arr[major_ch] if major_ch < arr.shape[0] else np.zeros(arr.shape[1:], dtype=np.float32)
            img_list.append(ch_img.astype(np.float32))
            meta_rows.append({'frame': i, 'group': gname})
        images_raw_arr = np.stack(img_list)
        img_meta_df    = pd.DataFrame(meta_rows)
        print(f"[pack] Built {len(img_list)} canvas images from CZI ch{major_ch} "
              f"({images_raw_arr.shape[-2]}×{images_raw_arr.shape[-1]}px)")

    # ── 6. Write HDF5 ─────────────────────────────────────────────────────────
    print(f"[pack] Writing {out_h5} …")
    with h5py.File(out_h5, 'w') as f:
        f.create_dataset('meta/csv', data=df.to_csv(index=False).encode('utf-8'))
        f.attrs['pad_size']    = pad
        f.attrs['image_scale'] = args.image_scale
        f.attrs['result_dir']  = str(plot_dir)
        f.attrs['n_patches']   = len(df)

        if patches_raw_arr is not None:
            f.create_dataset('patches/raw', data=patches_raw_arr,
                             compression='gzip', compression_opts=4)

        if allch_arr is not None:
            f.create_dataset('patches/allch', data=allch_arr,
                             compression='gzip', compression_opts=4)
            f.attrs['n_channels'] = int(allch_arr.shape[1])
            if channel_names_list:
                f.attrs['channel_names'] = json.dumps(channel_names_list)

        if allch_images:
            g = f.create_group('images/allch')
            for gname, arr in allch_images.items():
                g.create_dataset(gname, data=arr,
                                 compression='gzip', compression_opts=4)
            print(f"[pack] Wrote images/allch: {len(allch_images)} groups")

        if images_raw_arr is not None:
            f.create_dataset('images/raw',  data=images_raw_arr,
                             compression='gzip', compression_opts=4)
            f.create_dataset('images/meta',
                             data=img_meta_df.to_csv(index=False).encode('utf-8'))

    size_mb = out_h5.stat().st_size / 1e6
    print(f"[pack] Done — {out_h5.name}  ({size_mb:.1f} MB,  {len(df)} patches)")


if __name__ == '__main__':
    main()
