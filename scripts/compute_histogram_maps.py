#!/usr/bin/env python3
"""
Compute histogram matching maps for cross-dataset normalisation.

Steps:
  1. Sample patches from all datasets (vinc, ppax, pfak, nih3t3).
  2. Pool all pixels → compute reference CDF quantiles.
  3. Per dataset: compute source CDF quantiles.
  4. Save forward map  (src → ref)  and inverse map  (ref → src)
     as a .npz file per dataset under --out-dir.

Output files:
  <out_dir>/reference_quantiles.npy    — shared reference CDF  (N_Q,)
  <out_dir>/vinc_map.npz               — keys: src_q, ref_q
  <out_dir>/ppax_map.npz
  <out_dir>/pfak_map.npz
  <out_dir>/nih3t3_map.npz

Forward  apply:  np.interp(patch, src_q, ref_q)
Inverse  apply:  np.interp(patch, ref_q, src_q)

Usage:
  python scripts/compute_histogram_maps.py
  python scripts/compute_histogram_maps.py --n-patches 5000 --n-quantiles 10000
"""

import argparse
import random
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path("/net/projects/CLS/lding/data/fa_data_analysis")
BASE = ROOT / "ae_results/pax_ch_patch/cio_rb"

DATASET_DIRS = {
    "vinc":   [BASE/"vinc/control/tiff_patches32",  BASE/"vinc/ycomp/tiff_patches32"],
    "ppax":   [BASE/"ppax/control/tiff_patches32",  BASE/"ppax/ycomp/tiff_patches32"],
    "pfak":   [BASE/"pfak/control/tiff_patches32",  BASE/"pfak/ycomp/tiff_patches32"],
    "nih3t3": [BASE/"nih3t3/control/tiff_patches32",BASE/"nih3t3/ycomp/tiff_patches32"],
}


def _sample_pixels(dirs, n_patches, rng):
    files = []
    for d in dirs:
        files += list(Path(d).glob("*.tif")) + list(Path(d).glob("*.tiff"))
    chosen = rng.sample(files, min(n_patches, len(files)))
    pixels = []
    for f in chosen:
        pixels.append(tifffile.imread(str(f)).astype(np.float32).ravel())
    return np.concatenate(pixels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir",    type=Path,
                        default=ROOT / "ae_results/histogram_maps")
    parser.add_argument("--n-patches",  type=int, default=3000,
                        help="Patches sampled per dataset")
    parser.add_argument("--n-quantiles",type=int, default=10000,
                        help="Number of quantile points in lookup table")
    parser.add_argument("--ref-scale",  type=float, default=1.0,
                        help="Divide reference quantiles by this factor before saving "
                             "(e.g. 3 to compress the mapped range toward 0~1)")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng  = random.Random(args.seed)
    q    = np.linspace(0, 1, args.n_quantiles)

    # ── step 1: collect pixels per dataset ───────────────────────────────────
    ds_pixels = {}
    for ds, dirs in DATASET_DIRS.items():
        print(f"Sampling {ds} …", flush=True)
        pix = _sample_pixels(dirs, args.n_patches, rng)
        ds_pixels[ds] = pix
        print(f"  {len(pix):,} pixels  mean={pix.mean():.4f}  std={pix.std():.4f}")

    # ── step 2: reference quantiles from all datasets pooled ─────────────────
    all_pixels = np.concatenate(list(ds_pixels.values()))
    ref_q      = np.quantile(all_pixels, q).astype(np.float32)
    np.save(str(args.out_dir / "reference_quantiles.npy"), ref_q)
    print(f"\nReference: {len(all_pixels):,} pixels  "
          f"mean={all_pixels.mean():.4f}  std={all_pixels.std():.4f}")
    print(f"Saved reference_quantiles.npy")

    # ── step 3: per-dataset forward + inverse maps ────────────────────────────
    for ds, pix in ds_pixels.items():
        src_q = np.quantile(pix, q).astype(np.float32)
        out   = args.out_dir / f"{ds}_map.npz"
        np.savez(str(out), src_q=src_q, ref_q=ref_q)
        # quick sanity: check mapping endpoints
        print(f"{ds:8s}: src_q[0]={src_q[0]:.4f}→{ref_q[0]:.4f}  "
              f"src_q[-1]={src_q[-1]:.4f}→{ref_q[-1]:.4f}  saved {out.name}")

    print(f"\nAll maps saved to {args.out_dir}")
    print("\nUsage:")
    print("  Forward:  np.interp(patch, src_q, ref_q)   # standardise to reference")
    print("  Inverse:  np.interp(patch, ref_q, src_q)   # recover original appearance")


if __name__ == "__main__":
    main()
