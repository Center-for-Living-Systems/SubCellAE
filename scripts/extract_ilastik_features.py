#!/usr/bin/env python3
"""
extract_ilastik_features.py

Extract ilastik-style multiscale filter-bank features from 32-px FA patch TIFFs.

Features (80 total = 40 feature maps × 2 summary stats)
---------------------------------------------------------
Filter types (6) × 5 scales each → 40 feature maps:
  1. Gaussian smoothing
  2. Laplacian of Gaussian (LoG)
  3. Gaussian gradient magnitude
  4. Difference of Gaussians (DoG, σ vs σ×√2)
  5. Structure tensor — larger eigenvalue
  6. Structure tensor — smaller eigenvalue
  (Hessian eigenvalues merged into structure tensor slot for clean 40-map count)

Scales: [0.3, 0.7, 1.0, 1.6, 3.5] pixels

Summary stats per feature map: mean, std → 80 features total

Output
------
  DATA/ae_results/features/ilastik/{ds}.csv
  Columns: filename, <80 feature cols>

Usage
-----
  python scripts/extract_ilastik_features.py --dataset ds1
  python scripts/extract_ilastik_features.py --all
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import gaussian_filter, gaussian_laplace

DATA       = Path("/net/projects/CLS/lding/data/fa_data_analysis")
PATCH_ROOT = DATA / "ae_results/patches/cio"
OUT_ROOT   = DATA / "ae_results/features/ilastik"

PATCH_DIRS = {
    "ds1": [PATCH_ROOT / "vinc/control/tiff_patches32_mr10",
            PATCH_ROOT / "vinc/ycomp/tiff_patches32_mr10"],
    "ds2": [PATCH_ROOT / "pfak/control/tiff_patches32_mr10",
            PATCH_ROOT / "pfak/ycomp/tiff_patches32_mr10"],
    "ds3": [PATCH_ROOT / "ppax/control/tiff_patches32_mr10",
            PATCH_ROOT / "ppax/ycomp/tiff_patches32_mr10"],
}

SCALES = [0.3, 0.7, 1.0, 1.6, 3.5]

FILTER_TYPES = [
    "gaussian",
    "log",
    "gradient_mag",
    "dog",
    "structure_tensor_large",
    "structure_tensor_small",
    "hessian_large",
    "hessian_small",
]

# Build feature names: {filter}_{stat}_s{scale}
FEATURE_NAMES = []
for ftype in FILTER_TYPES:
    for sigma in SCALES:
        s = str(sigma).replace(".", "p")
        for stat in ["mean", "std"]:
            FEATURE_NAMES.append(f"{ftype}_{stat}_s{s}")


# ---------------------------------------------------------------------------
# Filter computations

def _gaussian_derivatives(img: np.ndarray, sigma: float):
    """Return (Gx, Gy) — first-order Gaussian derivatives."""
    Gx = gaussian_filter(img, sigma=sigma, order=[0, 1])
    Gy = gaussian_filter(img, sigma=sigma, order=[1, 0])
    return Gx, Gy


def _hessian(img: np.ndarray, sigma: float):
    """Return (Hxx, Hxy, Hyy) — second-order Gaussian derivatives."""
    Hxx = gaussian_filter(img, sigma=sigma, order=[0, 2])
    Hxy = gaussian_filter(img, sigma=sigma, order=[1, 1])
    Hyy = gaussian_filter(img, sigma=sigma, order=[2, 0])
    return Hxx, Hxy, Hyy


def _sym2x2_eigenvalues(A, B, C):
    """
    Larger and smaller eigenvalues of [[A, B], [B, C]] per pixel.
    A, B, C: (H, W) arrays.
    """
    trace  = A + C
    disc   = np.sqrt(np.maximum((A - C) ** 2 + 4 * B ** 2, 0))
    lam1   = (trace + disc) / 2   # larger
    lam2   = (trace - disc) / 2   # smaller
    return lam1, lam2


def _agg(arr: np.ndarray) -> tuple[float, float]:
    """Return (mean, std) of a feature map."""
    return float(arr.mean()), float(arr.std())


def ilastik_features(img: np.ndarray) -> np.ndarray:
    """
    img: (H, W) float32.
    Returns (80,) float32 feature vector.
    """
    feats = []

    for sigma in SCALES:
        # 1. Gaussian
        g = gaussian_filter(img, sigma=sigma)
        feats.extend(_agg(g))

    for sigma in SCALES:
        # 2. Laplacian of Gaussian
        log = gaussian_laplace(img, sigma=sigma)
        feats.extend(_agg(log))

    for sigma in SCALES:
        # 3. Gradient magnitude
        Gx, Gy = _gaussian_derivatives(img, sigma)
        grad_mag = np.sqrt(Gx ** 2 + Gy ** 2)
        feats.extend(_agg(grad_mag))

    for sigma in SCALES:
        # 4. DoG: Gaussian(σ) − Gaussian(σ × √2)
        dog = gaussian_filter(img, sigma=sigma) - gaussian_filter(img, sigma=sigma * np.sqrt(2))
        feats.extend(_agg(dog))

    for sigma in SCALES:
        # 5. Structure tensor — larger eigenvalue
        Gx, Gy = _gaussian_derivatives(img, sigma)
        Jxx = gaussian_filter(Gx * Gx, sigma=sigma)
        Jxy = gaussian_filter(Gx * Gy, sigma=sigma)
        Jyy = gaussian_filter(Gy * Gy, sigma=sigma)
        lam1, _ = _sym2x2_eigenvalues(Jxx, Jxy, Jyy)
        feats.extend(_agg(lam1))

    for sigma in SCALES:
        # 6. Structure tensor — smaller eigenvalue
        Gx, Gy = _gaussian_derivatives(img, sigma)
        Jxx = gaussian_filter(Gx * Gx, sigma=sigma)
        Jxy = gaussian_filter(Gx * Gy, sigma=sigma)
        Jyy = gaussian_filter(Gy * Gy, sigma=sigma)
        _, lam2 = _sym2x2_eigenvalues(Jxx, Jxy, Jyy)
        feats.extend(_agg(lam2))

    for sigma in SCALES:
        # 7. Hessian — larger eigenvalue
        Hxx, Hxy, Hyy = _hessian(img, sigma)
        lam1, _ = _sym2x2_eigenvalues(Hxx, Hxy, Hyy)
        feats.extend(_agg(lam1))

    for sigma in SCALES:
        # 8. Hessian — smaller eigenvalue
        Hxx, Hxy, Hyy = _hessian(img, sigma)
        _, lam2 = _sym2x2_eigenvalues(Hxx, Hxy, Hyy)
        feats.extend(_agg(lam2))

    return np.array(feats, dtype=np.float32)


def extract_patch(path: Path) -> np.ndarray:
    img = tifffile.imread(str(path)).astype(np.float32)
    return ilastik_features(img)


# ---------------------------------------------------------------------------
# Main

def process_dataset(ds: str):
    patch_dirs = [d for d in PATCH_DIRS[ds] if d.exists()]
    all_tifs   = []
    for d in patch_dirs:
        all_tifs.extend(sorted(d.glob("*.tif")))

    print(f"{ds}: {len(all_tifs)} patches across {len(patch_dirs)} dirs")

    rows = []
    for i, p in enumerate(all_tifs):
        if i % 1000 == 0:
            print(f"  {i}/{len(all_tifs)} ...", flush=True)
        try:
            feat = extract_patch(p)
            rows.append([p.name] + feat.tolist())
        except Exception as e:
            print(f"  WARNING: skipping {p.name}: {e}")

    df = pd.DataFrame(rows, columns=["filename"] + FEATURE_NAMES)
    out = OUT_ROOT / f"{ds}.csv"
    df.to_csv(out, index=False)
    print(f"  Saved {len(df)} rows × {len(FEATURE_NAMES)} features → {out}")
    return df


def main():
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--dataset", choices=["ds1", "ds2", "ds3"])
    grp.add_argument("--all", action="store_true")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    datasets = ["ds1", "ds2", "ds3"] if args.all else [args.dataset]
    for ds in datasets:
        process_dataset(ds)

    print("\nFeature names:")
    for i, n in enumerate(FEATURE_NAMES):
        print(f"  {i+1:3d}. {n}")


if __name__ == "__main__":
    main()
