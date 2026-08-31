#!/usr/bin/env python3
"""
extract_cp_features.py

Extract CellProfiler-style handcrafted features from 32-px FA patch TIFFs.

Features (50 total)
-------------------
Intensity (11):
  mean, std, median, min, max, integrated (sum), MAD,
  p10, p25, p75, p90

Haralick/GLCM texture (13 per distance × 3 distances = 39):
  Distances: [1, 2, 4] pixels, averaged over 4 angles (0°/45°/90°/135°)
  Features per distance:
    angular_second_moment, contrast, correlation, variance,
    inverse_difference_moment, sum_average, sum_variance, sum_entropy,
    entropy, difference_variance, difference_entropy, info_meas1, info_meas2

Output
------
  DATA/ae_results/features/cellprofiler/{ds}.csv
  Columns: filename, <50 feature cols>

Usage
-----
  python scripts/extract_cp_features.py --dataset ds1
  python scripts/extract_cp_features.py --dataset ds2
  python scripts/extract_cp_features.py --dataset ds3
  python scripts/extract_cp_features.py --all
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage.feature import graycomatrix

DATA       = Path("/net/projects/CLS/lding/data/fa_data_analysis")
PATCH_ROOT = DATA / "ae_results/patches/cio"
OUT_ROOT   = DATA / "ae_results/features/cellprofiler"

PATCH_DIRS = {
    "ds1": [PATCH_ROOT / "vinc/control/tiff_patches32_mr10",
            PATCH_ROOT / "vinc/ycomp/tiff_patches32_mr10"],
    "ds2": [PATCH_ROOT / "pfak/control/tiff_patches32_mr10",
            PATCH_ROOT / "pfak/ycomp/tiff_patches32_mr10"],
    "ds3": [PATCH_ROOT / "ppax/control/tiff_patches32_mr10",
            PATCH_ROOT / "ppax/ycomp/tiff_patches32_mr10"],
}

GLCM_DISTANCES = [1, 2, 4]
GLCM_ANGLES    = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_LEVELS    = 16    # gray levels for GLCM (adequate for 32×32 patch)

INTENSITY_NAMES = [
    "intensity_mean", "intensity_std", "intensity_median",
    "intensity_min",  "intensity_max", "intensity_integrated",
    "intensity_mad",  "intensity_p10", "intensity_p25",
    "intensity_p75",  "intensity_p90",
]

HARALICK_NAMES = [
    "angular_second_moment", "contrast", "correlation", "variance",
    "inverse_difference_moment", "sum_average", "sum_variance", "sum_entropy",
    "entropy", "difference_variance", "difference_entropy",
    "info_meas1", "info_meas2",
]

FEATURE_NAMES = INTENSITY_NAMES + [
    f"{h}_d{d}" for d in GLCM_DISTANCES for h in HARALICK_NAMES
]


# ---------------------------------------------------------------------------
# Feature computation

def intensity_features(img: np.ndarray) -> np.ndarray:
    """img: (H, W) float32, any value range."""
    flat = img.ravel()
    mad  = np.median(np.abs(flat - np.median(flat)))
    return np.array([
        flat.mean(),
        flat.std(),
        np.median(flat),
        flat.min(),
        flat.max(),
        flat.sum(),
        mad,
        np.percentile(flat, 10),
        np.percentile(flat, 25),
        np.percentile(flat, 75),
        np.percentile(flat, 90),
    ], dtype=np.float32)


def _glcm_haralick(P: np.ndarray) -> np.ndarray:
    """
    Compute 13 Haralick features from a single normalised GLCM P (N×N).
    P must be normalised (sum=1) and symmetric.
    Definitions follow CellProfiler / mahotas convention.
    """
    N   = P.shape[0]
    eps = 1e-10
    i   = np.arange(N, dtype=np.float64)
    j   = np.arange(N, dtype=np.float64)
    I, J = np.meshgrid(i, j, indexing="ij")

    # Marginals
    px  = P.sum(axis=1)    # (N,)
    py  = P.sum(axis=0)    # (N,)

    # Sum and difference distributions
    # p_{x+y}(k) for k = 0 .. 2*(N-1)
    k_sum = np.arange(2 * N - 1)
    pxy   = np.zeros(2 * N - 1)
    for ki, k in enumerate(k_sum):
        mask = (I + J).astype(int) == k
        pxy[ki] = P[mask].sum()

    # p_{x-y}(k) for k = 0 .. N-1
    k_diff = np.arange(N)
    pxmy   = np.zeros(N)
    for ki, k in enumerate(k_diff):
        mask = np.abs(I - J).astype(int) == k
        pxmy[ki] = P[mask].sum()

    # Means and stds of marginals
    mu_x = (i * px).sum()
    mu_y = (j * py).sum()
    sg_x = np.sqrt(((i - mu_x) ** 2 * px).sum() + eps)
    sg_y = np.sqrt(((j - mu_y) ** 2 * py).sum() + eps)

    # HX, HY for info measures
    HX  = -(px[px > 0] * np.log2(px[px > 0] + eps)).sum()
    HY  = -(py[py > 0] * np.log2(py[py > 0] + eps)).sum()

    # 1. Angular Second Moment (Energy)
    f1 = (P ** 2).sum()

    # 2. Contrast
    f2 = ((I - J) ** 2 * P).sum()

    # 3. Correlation
    f3 = ((I * J * P).sum() - mu_x * mu_y) / (sg_x * sg_y + eps)

    # 4. Variance (sum of squares of px)
    f4 = ((i - mu_x) ** 2 * px).sum()

    # 5. Inverse Difference Moment
    f5 = (P / (1 + (I - J) ** 2)).sum()

    # 6. Sum Average
    f6 = (k_sum * pxy).sum()

    # 7. Sum Variance
    f7 = ((k_sum - f6) ** 2 * pxy).sum()

    # 8. Sum Entropy
    pxy_nz = pxy[pxy > 0]
    f8 = -(pxy_nz * np.log2(pxy_nz + eps)).sum()

    # 9. Entropy
    P_nz = P[P > 0]
    f9 = -(P_nz * np.log2(P_nz + eps)).sum()

    # 10. Difference Variance
    mu_diff = (k_diff * pxmy).sum()
    f10 = ((k_diff - mu_diff) ** 2 * pxmy).sum()

    # 11. Difference Entropy
    pxmy_nz = pxmy[pxmy > 0]
    f11 = -(pxmy_nz * np.log2(pxmy_nz + eps)).sum()

    # 12 & 13. Information Measures of Correlation
    # HXY = entropy of P (f9)
    # HXY1 = -sum p(i,j) log2(px(i)*py(j))
    pxpy = np.outer(px, py)
    pxpy_nz = pxpy[P > 0]
    P_nz2   = P[P > 0]
    HXY1 = -(P_nz2 * np.log2(pxpy_nz + eps)).sum()
    # HXY2 = -sum px(i)*py(j) log2(px(i)*py(j))
    pxpy_flat = pxpy_nz
    HXY2 = -(pxpy_flat * np.log2(pxpy_flat + eps)).sum()
    f12 = (HXY1 - f9) / (max(HX, HY) + eps)
    arg = max(0.0, 1.0 - np.exp(-2.0 * (HXY2 - f9)))
    f13 = np.sqrt(arg)

    return np.array([f1, f2, f3, f4, f5, f6, f7, f8,
                     f9, f10, f11, f12, f13], dtype=np.float32)


def haralick_features(img: np.ndarray) -> np.ndarray:
    """
    Compute Haralick features at 3 distances, averaged over 4 angles.
    img: (H, W) float32 — will be quantised to GLCM_LEVELS bins.
    Returns (39,) float32 array.
    """
    # Quantise to [0, GLCM_LEVELS-1] uint8
    lo, hi = img.min(), img.max()
    img_q  = ((img - lo) / (hi - lo + 1e-9) * (GLCM_LEVELS - 1)).astype(np.uint8)

    feats = []
    for d in GLCM_DISTANCES:
        # GLCM: shape (GLCM_LEVELS, GLCM_LEVELS, 1, n_angles)
        P_all = graycomatrix(img_q, distances=[d], angles=GLCM_ANGLES,
                             levels=GLCM_LEVELS, symmetric=True, normed=True)
        # Average over angles → (GLCM_LEVELS, GLCM_LEVELS)
        P_avg = P_all[:, :, 0, :].mean(axis=-1)
        feats.append(_glcm_haralick(P_avg))

    return np.concatenate(feats)


def extract_patch(path: Path) -> np.ndarray:
    """Load TIFF and return (50,) feature vector."""
    img = tifffile.imread(str(path)).astype(np.float32)
    return np.concatenate([intensity_features(img), haralick_features(img)])


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
        print(f"  {i+1:2d}. {n}")


if __name__ == "__main__":
    main()
