"""
dataset.py
==========
PyTorch Dataset for binary adhesion-vs-non-adhesion screening.

Each sample is a 32×32 single-channel TIFF patch.  The loader converts it to
a 3-channel float tensor (by repeating the channel), clips to [0, 1], and
applies ImageNet normalization so pretrained EfficientNet weights transfer
correctly.

Binary label encoding
---------------------
  1 → "adhesion"     (Nascent Adhesion / focal complex / focal adhesion / fibrillar adhesion)
  0 → "no adhesion"  (No adhesion)

Labels listed in *exclude_labels* (default: ["Uncertain"]) are dropped.

Dataset-level intensity corrections
-------------------------------------
The cio_rb normalization divides each patch by the per-cell mean-inside
intensity.  Because different proteins (vinculin vs ppax) have different
expression levels and spatial distributions, the resulting per-cell scale
can differ systematically between datasets even after cio_rb.

Two correction classes are provided:

``DatasetLinearCorrection``
    Matches only the first two moments (mean and std) of the source to the
    reference.  Fast, but does not correct distributional shape differences
    (skewness, bimodality, tail behaviour).

``DatasetHistogramCorrection``
    True histogram matching — maps the full empirical CDF of the source onto
    the reference CDF.  For each source pixel value *x*, finds the quantile
    *q = CDF_src(x)* and maps it to *CDF_ref⁻¹(q)*.  This is a monotonic
    nonlinear remapping that aligns the complete distribution shape while
    preserving intra-patch relative intensity orderings — so the biological
    signal (Nascent Adhesion is dim) is intact.

Use ``sample_dataset_pixels`` to collect a pixel array, then
``compute_histogram_correction`` to build the correction object.
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import tifffile
import torch
from torch.utils.data import Dataset
from torchvision import transforms

log = logging.getLogger(__name__)

# ImageNet statistics (applied after converting grayscale → 3-channel)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

AD_LABELS: tuple[str, ...] = (
    "Nascent Adhesion",
    "focal complex",
    "focal adhesion",
    "fibrillar adhesion",
)
NONAD_LABEL = "No adhesion"

LABEL_NAMES = ["no adhesion", "adhesion"]


def _uid_to_fname(uid: str) -> str:
    """Convert label-CSV unique_ID (hyphen) to patch filename (underscore)."""
    return uid.replace("-", "_", 1)


# ---------------------------------------------------------------------------
# Dataset-level linear correction
# ---------------------------------------------------------------------------

def compute_dataset_stats(
    patch_dirs: dict[str, str | Path] | list[str | Path],
    max_patches: Optional[int] = None,
) -> tuple[float, float]:
    """Compute pixel-level mean and std across all patches in the given directories.

    Loads every ``.tif`` file (or up to *max_patches* sampled uniformly) and
    returns the global mean and std over all pixels.  Values are read as-is
    from disk (no clipping) so the statistics reflect the true distribution.

    Parameters
    ----------
    patch_dirs : dict or list
        Either a mapping ``{condition: dir_path}`` (same format as
        ``PatchScreeningDataset``) or a plain list of directories.
    max_patches : int, optional
        If given, sample at most this many patches to keep runtime short.
        Shuffled uniformly so the sample is representative.

    Returns
    -------
    (mean, std) : (float, float)
        Population statistics over all sampled pixels.
    """
    if isinstance(patch_dirs, dict):
        dirs = list(patch_dirs.values())
    else:
        dirs = list(patch_dirs)

    all_files: list[Path] = []
    for d in dirs:
        all_files.extend(Path(d).glob("*.tif"))

    if not all_files:
        raise FileNotFoundError(f"No .tif files found in: {dirs}")

    if max_patches is not None and len(all_files) > max_patches:
        rng = np.random.default_rng(0)
        all_files = [all_files[i]
                     for i in rng.choice(len(all_files), max_patches, replace=False)]

    log.info("compute_dataset_stats: loading %d patches from %d dirs …",
             len(all_files), len(dirs))

    # Incremental mean/variance using numpy (vectorised, fast)
    # Uses the parallel / batch form of Welford to keep memory bounded:
    # accumulate (n, mean, M2) across patch batches.
    total_n  = 0
    total_mean = 0.0
    total_M2   = 0.0

    for fpath in all_files:
        pixels = tifffile.imread(str(fpath)).ravel().astype(np.float64)
        b_n    = len(pixels)
        b_mean = pixels.mean()
        b_M2   = pixels.var() * b_n          # sum of squared deviations

        # Combine with running accumulator (Chan et al. parallel formula)
        delta       = b_mean - total_mean
        new_n       = total_n + b_n
        total_mean += delta * b_n / new_n
        total_M2   += b_M2 + delta ** 2 * total_n * b_n / new_n
        total_n     = new_n

    std = float(np.sqrt(total_M2 / total_n)) if total_n > 1 else 1.0
    log.info("  → mean=%.5f  std=%.5f  (n=%d pixels)", total_mean, std, total_n)
    return float(total_mean), std


class DatasetLinearCorrection:
    """Linear pixel-value correction that maps source → reference distribution.

    The transformation is::

        x_out = (x_in - src_mean) / src_std * ref_std + ref_mean

    This is a global affine remap that aligns the two datasets' pixel
    distributions while preserving all intra-patch relative intensities
    (it is the same linear operation applied uniformly to every pixel).

    Parameters
    ----------
    ref_mean, ref_std : float
        Statistics of the *reference* (training) dataset.
    src_mean, src_std : float
        Statistics of the *source* (test/new) dataset to be corrected.
    """

    def __init__(
        self,
        ref_mean: float,
        ref_std: float,
        src_mean: float,
        src_std: float,
    ):
        self.ref_mean = ref_mean
        self.ref_std  = ref_std
        self.src_mean = src_mean
        self.src_std  = src_std

        # Pre-compute y = x * scale + shift
        self.scale = ref_std / (src_std + 1e-8)
        self.shift = ref_mean - src_mean * self.scale

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Apply correction to a float32 numpy array.  Returns float32."""
        return (img.astype(np.float64) * self.scale + self.shift).astype(np.float32)

    def __repr__(self) -> str:
        return (f"DatasetLinearCorrection("
                f"ref=({self.ref_mean:.4f}, {self.ref_std:.4f}), "
                f"src=({self.src_mean:.4f}, {self.src_std:.4f}), "
                f"scale={self.scale:.4f}, shift={self.shift:.4f})")


# ---------------------------------------------------------------------------
# Dataset-level histogram matching correction
# ---------------------------------------------------------------------------

def sample_dataset_pixels(
    patch_dirs: dict[str, str | Path] | list[str | Path],
    max_patches: Optional[int] = None,
    seed: int = 0,
) -> np.ndarray:
    """Return a 1-D float32 array of pixel values sampled from patch TIFFs.

    Parameters
    ----------
    patch_dirs : dict or list
        Same format as ``PatchScreeningDataset`` or a plain list of directories.
    max_patches : int, optional
        Cap on the number of patches loaded (uniform random sample, seed=0).
    """
    if isinstance(patch_dirs, dict):
        dirs = list(patch_dirs.values())
    else:
        dirs = list(patch_dirs)

    all_files: list[Path] = []
    for d in dirs:
        all_files.extend(Path(d).glob("*.tif"))
    if not all_files:
        raise FileNotFoundError(f"No .tif files found in: {dirs}")

    if max_patches is not None and len(all_files) > max_patches:
        rng = np.random.default_rng(seed)
        all_files = [all_files[i]
                     for i in rng.choice(len(all_files), max_patches, replace=False)]

    log.info("sample_dataset_pixels: loading %d patches …", len(all_files))
    pixels = np.concatenate(
        [tifffile.imread(str(f)).ravel() for f in all_files]
    ).astype(np.float32)
    log.info("  → %d pixels  mean=%.4f  std=%.4f", len(pixels), pixels.mean(), pixels.std())
    return pixels


def compute_histogram_correction(
    ref_pixels: np.ndarray,
    src_pixels: np.ndarray,
    n_quantiles: int = 2000,
) -> "DatasetHistogramCorrection":
    """Build a ``DatasetHistogramCorrection`` from two pixel arrays.

    Parameters
    ----------
    ref_pixels : np.ndarray
        Flat array of pixel values from the *reference* (training) dataset.
    src_pixels : np.ndarray
        Flat array of pixel values from the *source* (test/new) dataset.
    n_quantiles : int
        Number of quantile breakpoints used to approximate the CDFs.
        Higher values give finer mapping; 2 000 is sufficient for 32-bit TIFFs.
    """
    quantiles  = np.linspace(0.0, 1.0, n_quantiles)
    ref_values = np.quantile(ref_pixels, quantiles).astype(np.float32)
    src_values = np.quantile(src_pixels, quantiles).astype(np.float32)
    return DatasetHistogramCorrection(src_values, ref_values)


class DatasetHistogramCorrection:
    """True histogram matching: maps source CDF onto reference CDF.

    For each pixel value *x* in a source patch the mapping is::

        q = CDF_src(x)          # find quantile in source distribution
        y = CDF_ref^{-1}(q)     # look up same quantile in reference

    Implemented efficiently as a piecewise-linear interpolation over
    *n_quantiles* breakpoints, so the per-pixel cost is O(log n_quantiles).

    The mapping is **monotonic** — intra-patch relative brightness is fully
    preserved (dim Nascent Adhesion patches remain dim after correction).

    Parameters
    ----------
    src_values : np.ndarray, shape (n_quantiles,)
        Pixel values at each quantile in the *source* distribution.
        Produced by ``compute_histogram_correction``.
    ref_values : np.ndarray, shape (n_quantiles,)
        Pixel values at each quantile in the *reference* distribution.

    Usage
    -----
    Build once::

        ref_pix = sample_dataset_pixels(vinc_dirs, max_patches=5000)
        src_pix = sample_dataset_pixels(ppax_dirs, max_patches=5000)
        hm = compute_histogram_correction(ref_pix, src_pix)

    Then pass as ``pixel_correction`` to ``PatchScreeningDataset``.
    """

    def __init__(self, src_values: np.ndarray, ref_values: np.ndarray):
        # Ensure strictly increasing (required for np.interp)
        self._src = np.asarray(src_values, dtype=np.float32)
        self._ref = np.asarray(ref_values, dtype=np.float32)
        # Deduplicate any flat regions in the source CDF so interp is well-defined
        _, unique_idx = np.unique(self._src, return_index=True)
        self._src = self._src[unique_idx]
        self._ref = self._ref[unique_idx]

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Apply histogram matching to a float32 numpy array. Returns float32."""
        return np.interp(
            img.ravel(), self._src, self._ref
        ).reshape(img.shape).astype(np.float32)

    def __repr__(self) -> str:
        return (f"DatasetHistogramCorrection("
                f"n_quantiles={len(self._src)}, "
                f"src_range=[{self._src[0]:.3f}, {self._src[-1]:.3f}], "
                f"ref_range=[{self._ref[0]:.3f}, {self._ref[-1]:.3f}])")


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def build_transforms(input_size: int = 224, augment: bool = False) -> Callable:
    """Return a torchvision transform pipeline (applied after pixel_correction and clip).

    Parameters
    ----------
    input_size : int
        Target spatial resolution (EfficientNet-B0 expects 224).
    augment : bool
        If True, apply random horizontal/vertical flip and rotation.
    """
    ops = []
    if augment:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=180),
        ]
    ops += [
        transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ]
    return transforms.Compose(ops)


class IntensityJitter:
    """Pixel-level intensity jitter applied to a float32 numpy patch.

    Applies a random multiplicative scale followed by a random additive shift,
    both sampled independently per patch.  Applied **before** clipping to
    [0, 1] so it changes the effective dynamic range visible to the model.

    This simulates the inter-dataset and inter-channel intensity variation
    observed between vinculin (vinc) and paxillin (ppax) patches, making the
    model more robust to distribution shift without destroying the relative
    intra-patch brightness (which is biologically meaningful for NA detection).

    Parameters
    ----------
    scale_range : (float, float)
        Multiplicative factor sampled uniformly from this range.
        Default (0.5, 2.0) — halves or doubles the dynamic range.
    shift_std : float
        Standard deviation of the zero-mean Gaussian additive shift.
        Default 0.05 — small baseline shift.
    """

    def __init__(
        self,
        scale_range: tuple[float, float] = (0.5, 2.0),
        shift_std: float = 0.05,
    ):
        self.scale_lo, self.scale_hi = scale_range
        self.shift_std = shift_std

    def __call__(self, img: np.ndarray) -> np.ndarray:
        scale = np.random.uniform(self.scale_lo, self.scale_hi)
        shift = np.random.normal(0.0, self.shift_std)
        return (img * scale + shift).astype(np.float32)

    def __repr__(self) -> str:
        return (f"IntensityJitter(scale=[{self.scale_lo},{self.scale_hi}], "
                f"shift_std={self.shift_std})")


class GammaJitter:
    """Random gamma correction applied to a float32 numpy patch.

    Clips input to [0, 1] then applies ``img ** gamma`` with gamma sampled
    uniformly.  Gamma < 1 brightens mid-tones (compresses dark end); gamma > 1
    darkens them (compresses bright end).  Complements IntensityJitter (which
    applies only linear scale + shift) by distorting the histogram shape
    non-linearly — mimicking differences in staining efficiency and imaging
    settings across datasets.

    Parameters
    ----------
    gamma_range : (float, float)
        Exponent sampled uniformly from this range.  Default (0.4, 2.5).
    """

    def __init__(self, gamma_range: tuple[float, float] = (0.4, 2.5)):
        self.gamma_lo, self.gamma_hi = gamma_range

    def __call__(self, img: np.ndarray) -> np.ndarray:
        gamma = np.random.uniform(self.gamma_lo, self.gamma_hi)
        return np.clip(img, 0.0, 1.0) ** gamma

    def __repr__(self) -> str:
        return f"GammaJitter(gamma=[{self.gamma_lo},{self.gamma_hi}])"


# ---------------------------------------------------------------------------
# Jitter-crop augmentation (on-the-fly from source frames)
# ---------------------------------------------------------------------------

# Parses: control_f0002x0784y0624ps32.tif → (condition, frame_idx, cx_pad, cy_pad, ps)
_COORD_RE = re.compile(
    r"^([a-zA-Z0-9]+)_f(\d+)x(\d+)y(\d+)ps(\d+)(?:\.tif)?$"
)


class JitterCropAugmentation:
    """On-the-fly crop with random translation and rotation from source frames.

    Instead of always reading the same pre-cropped 32×32 TIFF, this loads the
    full-frame source image and re-crops around the FA centre with a small
    random shift and rotation each call.  This provides translation and
    free-angle rotation diversity that is impossible from static patches.

    Source frames must be pre-extracted CIO-RB normalised TIFFs named:
        ``{frame_dir}/{condition}_f{idx:04d}_{channel}.tif``

    Parameters
    ----------
    frame_dirs : dict[str, str | Path]
        Mapping condition name → directory of source frame TIFFs.
    channel : str
        Channel name as used in source frame filenames (e.g. ``"pax"``,
        ``"vinc"``).
    max_shift_px : int
        Maximum ±pixel translation in x and y.  Default 4.
    max_angle_deg : float
        Maximum ±rotation in degrees.  Default 15.0.
    patch_size : int
        Output patch side length in pixels.  Must match the pre-cropped size.
    pad_size : int
        Padding added during patchprep (encoded in patch filenames as the
        offset to subtract to recover source-frame coordinates).  Default 64.
    """

    def __init__(
        self,
        frame_dirs: dict[str, str | Path],
        channel: str = "pax",
        max_shift_px: int = 4,
        max_angle_deg: float = 15.0,
        patch_size: int = 32,
        pad_size: int = 64,
    ):
        self.frame_dirs    = {k: Path(v) for k, v in frame_dirs.items()}
        self.channel       = channel
        self.max_shift_px  = max_shift_px
        self.max_angle_deg = max_angle_deg
        self.patch_size    = patch_size
        self.pad_size      = pad_size
        self._frame_cache: dict[tuple, np.ndarray] = {}

    def _load_frame(self, condition: str, frame_idx: int) -> np.ndarray:
        key = (condition, frame_idx)
        if key not in self._frame_cache:
            frame_dir = self.frame_dirs.get(condition)
            if frame_dir is None:
                raise KeyError(f"JitterCropAugmentation: no frame_dir for condition '{condition}'")
            fname = f"{condition}_f{frame_idx:04d}_{self.channel}.tif"
            fpath = frame_dir / fname
            if not fpath.exists():
                raise FileNotFoundError(f"Source frame not found: {fpath}")
            self._frame_cache[key] = tifffile.imread(str(fpath)).astype(np.float32)
        return self._frame_cache[key]

    def __call__(self, patch_path: str) -> np.ndarray:
        """Return a jitter-cropped float32 patch from the source frame.

        Parameters
        ----------
        patch_path : str
            Path to the original pre-cropped patch TIFF (only the filename is
            used to parse coordinates; the file itself is never read).
        """
        from scipy.ndimage import rotate as _ndrotate

        fname = Path(patch_path).stem
        m = _COORD_RE.match(fname)
        if m is None:
            # Fallback: read the patch file directly if filename doesn't parse
            return tifffile.imread(patch_path).astype(np.float32)

        condition  = m.group(1)
        frame_idx  = int(m.group(2))
        cx_pad     = int(m.group(3))
        cy_pad     = int(m.group(4))
        ps         = int(m.group(5))

        # Source-frame centre (undo patchprep padding offset)
        cx = cx_pad - self.pad_size
        cy = cy_pad - self.pad_size

        # Random jitter
        dx    = np.random.randint(-self.max_shift_px, self.max_shift_px + 1)
        dy    = np.random.randint(-self.max_shift_px, self.max_shift_px + 1)
        angle = np.random.uniform(-self.max_angle_deg, self.max_angle_deg)
        cx_j  = cx + dx
        cy_j  = cy + dy

        frame = self._load_frame(condition, frame_idx)
        fh, fw = frame.shape[:2]

        # Context region large enough to rotate without black corners
        angle_rad = math.radians(abs(angle))
        ctx = int(math.ceil(ps * (math.cos(angle_rad) + math.sin(angle_rad)))) + 4
        ctx = ctx + (ctx % 2)   # round up to even
        ctx = max(ctx, ps)

        # Reflect-pad frame so edge patches don't go out of bounds
        pad = ctx
        frame_padded = np.pad(frame, pad, mode="reflect")

        # Extract context region (coords shift by pad)
        cx_p = cx_j + pad
        cy_p = cy_j + pad
        x0 = cx_p - ctx // 2
        y0 = cy_p - ctx // 2
        x0 = max(0, min(x0, frame_padded.shape[1] - ctx))
        y0 = max(0, min(y0, frame_padded.shape[0] - ctx))
        ctx_crop = frame_padded[y0:y0 + ctx, x0:x0 + ctx]

        # Rotate context region
        rotated = _ndrotate(ctx_crop, angle, order=1, mode="reflect", reshape=False)

        # Centre-crop to patch_size
        cy_r = rotated.shape[0] // 2
        cx_r = rotated.shape[1] // 2
        half = ps // 2
        patch = rotated[cy_r - half:cy_r + half, cx_r - half:cx_r + half]

        # Ensure correct shape (guard against off-by-one from odd ctx)
        if patch.shape != (ps, ps):
            patch = patch[:ps, :ps]

        return patch.astype(np.float32)

    def __repr__(self) -> str:
        return (f"JitterCropAugmentation(channel={self.channel}, "
                f"shift=±{self.max_shift_px}px, angle=±{self.max_angle_deg}°)")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PatchScreeningDataset(Dataset):
    """Load labelled patches for binary adhesion screening.

    Parameters
    ----------
    label_csv : str or Path
        Path to the label CSV (must contain *filename_col*, *label_col*,
        *condition_col*, and *group_col*).
    patch_dirs : dict[str, str | Path]
        Mapping from condition name (e.g. ``"control"``) to the directory
        of patch TIFFs for that condition.
    label_col : str
    filename_col : str
        Column containing unique_ID (hyphen-separated filename).
    condition_col : str
    group_col : str
        Column used for group-aware train/val split (e.g. ``"czi_filename"``).
    ad_labels : sequence of str
        Labels that map to class 1 (adhesion).
    nonad_label : str
        Label that maps to class 0 (no adhesion).
    exclude_labels : sequence of str
        Labels to drop from the dataset entirely.
    extra_patch_dirs : list of dict, optional
        Additional channel directories treated as separate samples with the
        **same** labels.  Each element is a ``{condition: dir_path}`` mapping
        exactly like *patch_dirs*.  For each labeled patch the dataset will
        yield one sample per channel (primary + each extra).  Use this to
        combine e.g. paxillin (ch1) and vinculin (ch3) patches from the
        same experiment.
    pixel_correction : DatasetLinearCorrection or DatasetHistogramCorrection, optional
        Applied to the raw float32 image **before** clipping to ``[0, 1]``.
    intensity_jitter : IntensityJitter, optional
        Pixel-level intensity augmentation applied **before** clipping.
        Applied after *pixel_correction*, before the clip and tensor
        conversion.  Only active when the dataset is used for training
        (pass ``None`` for validation/test).
    gamma_jitter : GammaJitter, optional
        Non-linear histogram shape augmentation.  Applied after
        *intensity_jitter* (clips to [0,1] internally).  Training only.
    transform : callable, optional
        Torchvision transform applied to the 3-channel float tensor after
        clipping.
    indices : sequence of int, optional
        If given, use only these row indices (for train/val subsetting).
    """

    def __init__(
        self,
        label_csv: str | Path,
        patch_dirs: dict[str, str | Path],
        *,
        label_col: str = "classification",
        filename_col: str = "unique_ID",
        condition_col: str = "condition",
        group_col: str = "czi_filename",
        ad_labels: Sequence[str] = AD_LABELS,
        nonad_label: str = NONAD_LABEL,
        exclude_labels: Sequence[str] = ("Uncertain",),
        extra_patch_dirs: Optional[list] = None,
        pixel_correction=None,
        intensity_jitter: Optional["IntensityJitter"] = None,
        gamma_jitter: Optional["GammaJitter"] = None,
        jitter_crop: Optional["JitterCropAugmentation"] = None,
        transform: Optional[Callable] = None,
        indices: Optional[Sequence[int]] = None,
    ):
        df = pd.read_csv(label_csv)

        if exclude_labels:
            df = df[~df[label_col].isin(exclude_labels)].reset_index(drop=True)

        ad_set = set(ad_labels)
        valid_mask = df[label_col].isin(ad_set) | (df[label_col] == nonad_label)
        df = df[valid_mask].reset_index(drop=True)

        df["_binary_label"] = df[label_col].apply(
            lambda x: 1 if x in ad_set else 0
        )

        patch_dirs = {k: Path(v) for k, v in patch_dirs.items()}

        def _resolve_path(row) -> str:
            cond = str(row[condition_col])
            fname = _uid_to_fname(str(row[filename_col]))
            d = patch_dirs.get(cond)
            if d is None:
                return ""
            return str(d / fname)

        df["_patch_path"] = df.apply(_resolve_path, axis=1)

        exists_mask = df["_patch_path"].apply(lambda p: Path(p).exists() if p else False)
        n_missing = (~exists_mask).sum()
        if n_missing > 0:
            import warnings
            warnings.warn(f"PatchScreeningDataset: {n_missing} patch files not found — skipped.")
        df = df[exists_mask].reset_index(drop=True)

        # 1. Apply indices to PRIMARY channel rows first.
        #    This selects the train or val patch locations before any channel
        #    expansion, so the split is always computed on a consistent set of
        #    patch locations (not doubled by extra channels).
        if indices is not None:
            df = df.iloc[list(indices)].reset_index(drop=True)

        # 2. Expand with extra channels (each becomes a separate sample with
        #    the same binary label).  Expansion happens AFTER the split so that
        #    both channels of the same patch location always land in the same
        #    split (train or val), not split across them.
        #    Val datasets should pass extra_patch_dirs=None so only the primary
        #    channel is used for clean evaluation.
        if extra_patch_dirs:
            extra_dfs = []
            for extra_dirs in extra_patch_dirs:
                extra_dirs_p = {k: Path(v) for k, v in extra_dirs.items()}

                def _resolve_extra(row, dirs=extra_dirs_p) -> str:
                    cond  = str(row[condition_col])
                    fname = _uid_to_fname(str(row[filename_col]))
                    d     = dirs.get(cond)
                    return str(d / fname) if d else ""

                alt_df = df.copy()
                alt_df["_patch_path"] = alt_df.apply(_resolve_extra, axis=1)
                ok = alt_df["_patch_path"].apply(
                    lambda p: Path(p).exists() if p else False
                )
                n_miss = (~ok).sum()
                if n_miss > 0:
                    import warnings
                    warnings.warn(
                        f"PatchScreeningDataset extra_patch_dirs: "
                        f"{n_miss} patches not found — skipped."
                    )
                extra_dfs.append(alt_df[ok].copy())

            df = pd.concat([df] + extra_dfs, ignore_index=True)

        self._df              = df
        self._label_col       = label_col
        self._group_col       = group_col
        self.pixel_correction  = pixel_correction
        self.intensity_jitter  = intensity_jitter
        self.gamma_jitter      = gamma_jitter
        self.jitter_crop       = jitter_crop
        self.transform         = transform
        self.label_names       = LABEL_NAMES

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def labels(self) -> np.ndarray:
        return self._df["_binary_label"].values

    @property
    def groups(self) -> np.ndarray:
        return self._df[self._group_col].values

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int):
        row = self._df.iloc[idx]

        # 1. Load image — jitter crop re-samples from source frame each call;
        #    otherwise read the pre-cropped patch TIFF directly.
        if self.jitter_crop is not None:
            img = self.jitter_crop(row["_patch_path"])
        else:
            img = tifffile.imread(row["_patch_path"]).astype(np.float32)

        # 2. Dataset-level distribution correction (preserves intra-patch contrast)
        if self.pixel_correction is not None:
            img = self.pixel_correction(img)

        # 3. Intensity jitter (training only; applied before clip so it changes dynamic range)
        if self.intensity_jitter is not None:
            img = self.intensity_jitter(img)

        # 3b. Gamma jitter — non-linear histogram distortion (clips to [0,1] internally)
        if self.gamma_jitter is not None:
            img = self.gamma_jitter(img)

        # 4. Clip to [0, 1] → expand to 3-channel tensor (C, H, W)
        img = np.clip(img, 0.0, 1.0)
        tensor = torch.from_numpy(img).unsqueeze(0).repeat(3, 1, 1)

        # 5. Resize + spatial augmentation + ImageNet normalisation
        if self.transform is not None:
            tensor = self.transform(tensor)

        label = int(row["_binary_label"])
        return tensor, label
