#!/usr/bin/env python3
"""
pack_model_h5.py
================
Pack one model result directory into model.h5 for the interactive viewer.

This is the model half of the two-file H5 design:
  data.h5   ← pack_data_h5.py   (images + raw patches, shared across models)
  model.h5  ← this script       (latents, UMAP, predictions, recon patches)

The interactive viewer (view_interactive.py) takes both files:
  python scripts/view_interactive.py --data data.h5 --model model.h5

Primary CSV (tried in order):
  {result_dir}/analysis/analysis_results.csv  ← preferred (has UMAP already)
  {result_dir}/latents_newdata.csv
  {result_dir}/latents.csv

FA predictions (tried in order, first found wins):
  {result_dir}/fa_cls_lat8/   predictions_all.csv or classification_results.csv
  {result_dir}/fa_cls_zproj/  …
  {result_dir}/fa_cls_zrecon/ …

Position predictions (same pattern, pos_cls_* subdirs):

Recon patches:
  {result_dir}/recon/patches_recon.tif + patches_index.csv

Analysis plots:
  {result_dir}/analysis/*.png

Output:
  {result_dir}/model.h5     (or --out path)

H5 layout
---------
  meta/csv          bytes (CSV)   — filename, lat_*/z_*/p_*, UMAP_1, UMAP_2,
                                    fa_pred, fa_prob_*, pos_pred, pos_prob_*
  patches/recon     float32 (N, H, W)
  plots/{name}      uint8 bytes   — PNG blobs
  attrs: pad_size, image_scale, result_dir, n_patches, model_name

Usage
-----
    python scripts/pack_model_h5.py <result_dir>
    python scripts/pack_model_h5.py <result_dir> --out /tmp/model.h5
    python scripts/pack_model_h5.py <result_dir> --pad-size 64
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import tifffile


# ── helpers ───────────────────────────────────────────────────────────────────

def _find_csv(result_dir: Path, subdir: str,
              names: list[str]) -> Path | None:
    """Return first existing CSV among names inside subdir."""
    d = result_dir / subdir
    if not d.is_dir():
        return None
    for name in names:
        p = d / name
        if p.exists():
            return p
    return None


def _merge_predictions(df: pd.DataFrame, csv_path: Path,
                       prefix: str) -> pd.DataFrame:
    """Merge a classification CSV into df.

    Handles both column conventions:
      predictions_all.csv  →  pred_label, proba_*
      classification_results.csv  →  pred_label, prob_*
    Renames to {prefix}_pred and {prefix}_prob_*.
    """
    pred = pd.read_csv(csv_path)
    prob_cols = ([c for c in pred.columns if c.startswith('proba_')] or
                 [c for c in pred.columns if c.startswith('prob_')])
    keep = ['filename', 'pred_label'] + prob_cols
    keep = [c for c in keep if c in pred.columns]
    sub  = pred[keep].copy()
    sub  = sub.rename(columns={'pred_label': f'{prefix}_pred'})
    sub  = sub.rename(columns={c: f'{prefix}_{c}' for c in prob_cols})
    n_before = len(df)
    df = df.merge(sub, on='filename', how='left')
    n_matched = df[f'{prefix}_pred'].notna().sum()
    print(f'[pack_model]   {prefix}: {n_matched}/{n_before} matched, '
          f'{sub[f"{prefix}_pred"].nunique()} classes  ← {csv_path.name}')
    return df


# ── main packer ───────────────────────────────────────────────────────────────

def pack_model(result_dir: Path, out_h5: Path, pad_size: int = 64) -> None:
    if not result_dir.is_dir():
        print(f'[pack_model] ERROR: {result_dir} is not a directory', file=sys.stderr)
        sys.exit(1)

    print(f'[pack_model] result_dir : {result_dir}')
    print(f'[pack_model] output     : {out_h5}')

    # ── 1. Primary metadata CSV ───────────────────────────────────────────────
    csv_candidates = [
        result_dir / 'analysis' / 'analysis_results.csv',
        result_dir / 'latents_newdata.csv',
        result_dir / 'latents.csv',
    ]
    primary_csv = next((p for p in csv_candidates if p.exists()), None)
    if primary_csv is None:
        print('[pack_model] ERROR: no latents/analysis CSV found.', file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(primary_csv)
    print(f'[pack_model] {len(df)} patches from {primary_csv.name}')

    # ── 2. UMAP (merge separately if not already present) ────────────────────
    if 'UMAP_1' not in df.columns:
        # 2a. Try analysis_results.csv
        ana_csv = result_dir / 'analysis' / 'analysis_results.csv'
        if ana_csv.exists() and ana_csv != primary_csv:
            ana = pd.read_csv(ana_csv)
            umap_cols = [c for c in ana.columns if 'UMAP' in c.upper()]
            if len(umap_cols) >= 2:
                rename = {umap_cols[0]: 'UMAP_1', umap_cols[1]: 'UMAP_2'}
                df = df.merge(
                    ana[['filename'] + umap_cols[:2]].rename(columns=rename),
                    on='filename', how='left',
                )
                print(f'[pack_model] UMAP merged from analysis_results.csv')
            else:
                print(f'[pack_model] WARN: no UMAP columns in analysis_results.csv')

        # 2b. Try a saved UMAP model (.pkl) — transform z_* latents
        if 'UMAP_1' not in df.columns:
            UMAP_PKL_DIRS = ['fa_cls_lat8', 'fa_cls_zproj', 'fa_cls_zrecon',
                             'vis_lat8', 'vis_lat8dist8', 'analysis']
            z_cols = [c for c in df.columns if c.startswith('z_')]
            umap_pkl = next(
                (result_dir / d / 'umap_all_model.pkl'
                 for d in UMAP_PKL_DIRS
                 if (result_dir / d / 'umap_all_model.pkl').exists()),
                None,
            )
            if umap_pkl and z_cols:
                try:
                    import joblib
                    umap_model = joblib.load(str(umap_pkl))
                    z_arr = df[z_cols].values.astype(np.float32)
                    coords = umap_model.transform(z_arr)
                    df['UMAP_1'] = coords[:, 0]
                    df['UMAP_2'] = coords[:, 1]
                    print(f'[pack_model] UMAP computed via {umap_pkl.parent.name}/umap_all_model.pkl')
                except Exception as exc:
                    print(f'[pack_model] WARN: UMAP model transform failed: {exc}')
            else:
                print(f'[pack_model] WARN: UMAP not available'
                      + ('' if z_cols else ' (no z_* latent columns)'))
    else:
        print(f'[pack_model] UMAP_1/UMAP_2 already in primary CSV')

    # ── 3. FA-type predictions ────────────────────────────────────────────────
    FA_SUBDIRS  = ['fa_cls_lat8', 'fa_cls_zproj', 'fa_cls_zrecon']
    PRED_NAMES  = ['predictions_all.csv', 'classification_results.csv']
    fa_csv = next(
        (p for sd in FA_SUBDIRS
         for p in [_find_csv(result_dir, sd, PRED_NAMES)] if p),
        None,
    )
    if fa_csv:
        df = _merge_predictions(df, fa_csv, 'fa')
    else:
        print(f'[pack_model] WARN: no FA predictions found '
              f'(looked in {FA_SUBDIRS})')

    # ── 4. Position predictions ───────────────────────────────────────────────
    POS_SUBDIRS = ['pos_cls_lat8', 'pos_cls_zproj', 'pos_cls_zrecon']
    pos_csv = next(
        (p for sd in POS_SUBDIRS
         for p in [_find_csv(result_dir, sd, PRED_NAMES)] if p),
        None,
    )
    if pos_csv:
        df = _merge_predictions(df, pos_csv, 'pos')
    else:
        print(f'[pack_model] WARN: no position predictions found '
              f'(looked in {POS_SUBDIRS})')

    # ── 5. Reconstructed patches ──────────────────────────────────────────────
    recon_dir   = result_dir / 'recon'
    recon_tif   = recon_dir / 'patches_recon.tif'
    recon_idx   = recon_dir / 'patches_index.csv'
    patches_recon_arr = None

    recon_patch_dir = recon_dir / 'patches'

    if recon_tif.exists() and recon_idx.exists():
        # New stacked format: single TIF + index CSV
        patch_idx   = pd.read_csv(recon_idx)
        recon_stack = tifffile.imread(str(recon_tif)).astype(np.float32)
        # Shape may be (N, H, W) or (N, C, H, W) for multi-channel models — take ch0
        if recon_stack.ndim == 4:
            recon_stack = recon_stack[:, 0]
        _, H, W     = recon_stack.shape[:3]
        name_to_frame = {str(r['name']): int(r['frame'])
                         for _, r in patch_idx.iterrows()}
        stems = df['filename'].apply(lambda f: Path(str(f)).stem)
        n = len(df)
        patches_recon_arr = np.zeros((n, H, W), dtype=np.float32)
        missing = 0
        for i, stem in enumerate(stems):
            fi = name_to_frame.get(stem)
            if fi is not None:
                patches_recon_arr[i] = recon_stack[fi]
            else:
                missing += 1
        if missing:
            print(f'[pack_model] WARN: {missing}/{n} patches not in patches_index.csv')
        print(f'[pack_model] Recon patches: {n} × {H}×{W}px')

    elif recon_patch_dir.is_dir():
        # Old format: individual TIFFs named recon_{split}_{stem}.tif or recon_{stem}.tif
        stems  = df['filename'].apply(lambda f: Path(str(f)).stem)
        splits = (df['split'].fillna('').astype(str)
                  if 'split' in df.columns else pd.Series([''] * len(df)))
        n = len(df)

        def _recon_candidates(stem: str, split: str) -> list[Path]:
            cands = []
            if split:
                cands.append(recon_patch_dir / f'recon_{split}_{stem}.tif')
            cands.append(recon_patch_dir / f'recon_{stem}.tif')
            return cands

        # Detect H, W from first found file
        sample: Path | None = next(
            (p for s, sp in zip(stems, splits)
               for p in _recon_candidates(s, sp) if p.exists()),
            None,
        )
        if sample is None:
            print(f'[pack_model] WARN: no recon patches found in {recon_patch_dir}')
        else:
            _s = tifffile.imread(str(sample)).astype(np.float32)
            H, W = _s.shape[-2], _s.shape[-1]
            patches_recon_arr = np.zeros((n, H, W), dtype=np.float32)
            missing = 0
            for i, (stem, split) in enumerate(zip(stems, splits)):
                loaded = False
                for p in _recon_candidates(stem, split):
                    if p.exists():
                        arr = tifffile.imread(str(p)).astype(np.float32)
                        patches_recon_arr[i] = arr if arr.ndim == 2 else arr[0]
                        loaded = True
                        break
                if not loaded:
                    missing += 1
            if missing:
                print(f'[pack_model] WARN: {missing}/{n} recon patches not found')
            print(f'[pack_model] Recon patches (old format): {n-missing}/{n} × {H}×{W}px')

    else:
        print(f'[pack_model] WARN: recon TIFFs not found in {recon_dir} — no recon patches')

    # ── 6. Write HDF5 ─────────────────────────────────────────────────────────
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(out_h5), 'w') as f:
        f.create_dataset('meta/csv', data=np.bytes_(df.to_csv(index=False)))

        if patches_recon_arr is not None:
            f.create_dataset('patches/recon', data=patches_recon_arr,
                             compression='gzip', compression_opts=4)

        analysis_dir = result_dir / 'analysis'
        n_plots = 0
        for png in sorted(analysis_dir.glob('*.png')):
            f.create_dataset(f'plots/{png.stem}',
                             data=np.frombuffer(png.read_bytes(), dtype=np.uint8))
            n_plots += 1
        if n_plots:
            print(f'[pack_model] Packed {n_plots} analysis plots')

        f.attrs['pad_size']    = float(pad_size)
        f.attrs['image_scale'] = 1.0
        f.attrs['result_dir']  = str(result_dir)
        f.attrs['n_patches']   = int(len(df))
        f.attrs['model_name']  = result_dir.name

    size_mb = out_h5.stat().st_size / 1e6
    print(f'[pack_model] → {out_h5}  ({size_mb:.1f} MB,  {len(df)} patches)')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('result_dir',
                    help='Model result directory')
    ap.add_argument('--out', default=None,
                    help='Output path (default: result_dir/model.h5)')
    ap.add_argument('--pad-size', type=int, default=64,
                    help='Pad size used during patchprep (default: 64)')
    args = ap.parse_args()

    result_dir = Path(args.result_dir)
    out_h5     = Path(args.out) if args.out else result_dir / 'model.h5'
    pack_model(result_dir, out_h5, pad_size=args.pad_size)


if __name__ == '__main__':
    main()
