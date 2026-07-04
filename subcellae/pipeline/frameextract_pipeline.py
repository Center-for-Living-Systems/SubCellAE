"""
frameextract_pipeline.py
========================
Extract full-frame, per-channel normalized images from .czi files and save as TIFF.

For each source file the pipeline:
  1. Loads the raw CZI (all channels, values in [0, 1] after / 255²).
  2. Computes a cell segmentation mask (pre-computed file or on-the-fly).
  3. For each requested channel:
       a. Applies rolling-ball background subtraction (optional, radius in px).
       b. Applies cell-inside/outside (CIO) normalization with a per-channel
          scale constant:  out = (img - bg_mean) / (cell_mean - bg_mean) / scale
  4. Saves the normalized float32 image as a TIFF named
        {condition}_f{frame_idx:04d}_{channel_name}.tif

These full-frame images are the source for on-the-fly jitter-crop datasets that
randomly crop 32×32 patches at training time instead of loading pre-extracted
fixed patches.

Output filename format matches the patch-prep convention so that frame indices
are consistent: ``control_f0002_pax.tif`` contains the frame whose patches are
named ``control_f0002x????y????ps32.tif``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import tifffile

import subcellae.dataprep.patch_prep as patch_prep

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-channel config
# ---------------------------------------------------------------------------

@dataclass
class ChannelExtractConfig:
    """Configuration for one channel to extract.

    Parameters
    ----------
    index : int
        0-based channel index in the CZI file.
    name : str
        Short human-readable label used in the output filename (e.g. ``"pax"``).
    scale : float
        Divisor applied in the CIO normalization step.
        ``out = (img - bg_mean) / (cell_mean - bg_mean) / scale``.
        Larger values compress the dynamic range (fewer bright pixels above 1).
        Previously 5.0 for paxillin; increase to e.g. 8–10 for more headroom.
    """
    index: int
    name: str
    scale: float = 5.0


# ---------------------------------------------------------------------------
# Pipeline config
# ---------------------------------------------------------------------------

@dataclass
class FrameExtractConfig:
    """All parameters for one frame-extraction run.

    Parameters
    ----------
    image_folder : str
        Directory containing the raw .czi files.
    output_dir : str
        Directory where normalized TIFF frames are written (created if absent).
    condition : str
        Short label for the experimental condition (``"control"`` or ``"ycomp"``).
        Used as the prefix in output filenames.
    channels : list[ChannelExtractConfig]
        Channels to extract and normalize.  Each entry independently sets the
        channel index, name, and CIO scale constant.
    cell_mask_folder : str or None
        Directory with pre-computed cell-mask TIFFs
        (naming: ``cell_mask_<czi_basename>.tif``).
        ``None`` triggers on-the-fly segmentation.
    rolling_ball_radius : float or None
        Rolling-ball background subtraction radius (pixels) applied to each
        channel before CIO normalization.  ``None`` skips this step.
    seg_ch : int or None
        Channel index used for on-the-fly segmentation.
        ``None`` → uses the index of the first entry in *channels*.
    file_type : str
        ``"czi"`` (default) or ``"npy"``.
    start_ind : int
        First file index to process (inclusive).
    end_ind : int
        Last file index to process (exclusive).  Values beyond the number of
        files are clamped automatically.
    seg_threshold : float
        Binarization threshold for on-the-fly segmentation. Default ``0.1``.
    seg_close_size : int
        Closing disk radius (px). Default ``11``.
    seg_min_size_initial : int
        Minimum object area (px²) after initial threshold. Default ``3``.
    seg_min_size_post_close : int
        Minimum area after closing. Default ``10``.
    seg_min_size_final : int
        Minimum whole-cell area to retain. Default ``30000``.
    debug : bool
        If ``True``, stop after the first file.
    """
    image_folder: str
    output_dir: str
    condition: str
    channels: List[ChannelExtractConfig]

    cell_mask_folder: Optional[str] = None
    rolling_ball_radius: Optional[float] = None
    seg_ch: Optional[int] = None
    file_type: str = "czi"
    start_ind: int = 0
    end_ind: int = 999

    seg_threshold: float = 0.1
    seg_close_size: int = 11
    seg_min_size_initial: int = 3
    seg_min_size_post_close: int = 10
    seg_min_size_final: int = 30000

    debug: bool = False

    def __post_init__(self):
        if not self.channels:
            raise ValueError("At least one channel must be specified.")
        valid_file_types = {"czi", "npy"}
        if self.file_type not in valid_file_types:
            raise ValueError(
                f"file_type must be one of {valid_file_types}, got {self.file_type!r}"
            )
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def _compute_seg(
    raw: np.ndarray,
    filename: str,
    cfg: FrameExtractConfig,
) -> np.ndarray:
    """Return a 2-D float segmentation mask for one image.

    Uses a pre-computed mask file if ``cfg.cell_mask_folder`` is set and
    populated; falls back to on-the-fly segmentation otherwise.
    """
    _use_file_mask = (
        cfg.cell_mask_folder is not None
        and os.path.isdir(cfg.cell_mask_folder)
        and any(os.scandir(cfg.cell_mask_folder))
    )
    if _use_file_mask:
        mask_path = os.path.join(cfg.cell_mask_folder, "cell_mask_" + filename + ".tif")
        return tifffile.imread(mask_path).squeeze().astype(float)

    seg_ch = cfg.seg_ch if cfg.seg_ch is not None else cfg.channels[0].index
    seg_input = patch_prep._extract_channel(raw, seg_ch, filename, cfg.file_type)
    return patch_prep.segment_cell_mask(
        seg_input,
        threshold=cfg.seg_threshold,
        close_size=cfg.seg_close_size,
        min_size_initial=cfg.seg_min_size_initial,
        min_size_post_close=cfg.seg_min_size_post_close,
        min_size_final=cfg.seg_min_size_final,
    ).astype(float)


def _process_file(
    frame_idx: int,
    filename: str,
    cfg: FrameExtractConfig,
    output_dir: Path,
) -> None:
    """Extract, normalize, and save all channels for one source file."""
    raw = patch_prep._load_raw_squeezed(cfg.image_folder, filename, cfg.file_type)
    seg = _compute_seg(raw, filename, cfg)

    for ch_cfg in cfg.channels:
        img = patch_prep._extract_channel(raw, ch_cfg.index, filename, cfg.file_type)

        # Rolling-ball background subtraction
        if cfg.rolling_ball_radius is not None:
            if cfg.file_type == "czi":
                _scale = 255.0 * 255.0
                img = patch_prep.apply_rolling_ball(
                    img * _scale, radius=cfg.rolling_ball_radius
                ) / _scale
            else:
                img = patch_prep.apply_rolling_ball(img, radius=cfg.rolling_ball_radius)

        # CIO normalization with per-channel scale constant
        img = patch_prep.normalize_cell_insideoutside(img, seg, scale=ch_cfg.scale)

        out_path = output_dir / f"{cfg.condition}_f{frame_idx:04d}_{ch_cfg.name}.tif"
        tifffile.imwrite(
            str(out_path),
            img.astype(np.float32),
            imagej=True,
            metadata={"axes": "YX"},
        )
        log.debug("    saved %s", out_path.name)


def run_frameextract_pipeline(cfg: FrameExtractConfig) -> None:
    """Run the full frame-extraction pipeline.

    Parameters
    ----------
    cfg : FrameExtractConfig
        Fully-initialised configuration object.
    """
    output_dir = Path(cfg.output_dir)

    ch_summary = ", ".join(
        f"ch{c.index}={c.name}(scale={c.scale})" for c in cfg.channels
    )
    log.info("=" * 60)
    log.info("Frame Extract Pipeline")
    log.info("  image_folder     : %s", cfg.image_folder)
    log.info("  output_dir       : %s", cfg.output_dir)
    log.info("  condition        : %s", cfg.condition)
    log.info("  channels         : %s", ch_summary)
    log.info("  rolling_ball_r   : %s", cfg.rolling_ball_radius)
    log.info("  cell_mask_folder : %s", cfg.cell_mask_folder or "(on-the-fly seg)")
    log.info("  files [%d, %d)", cfg.start_ind, cfg.end_ind)
    log.info("=" * 60)

    filenames = patch_prep.list_image_files(cfg.image_folder, file_type=cfg.file_type)
    if not filenames:
        log.warning("No files found in %s", cfg.image_folder)
        return

    effective_end = min(cfg.end_ind, len(filenames))
    log.info(
        "Found %d file(s); processing indices %d–%d.",
        len(filenames), cfg.start_ind, effective_end - 1,
    )

    for frame_idx in range(cfg.start_ind, effective_end):
        filename = filenames[frame_idx]
        log.info(
            "  [%d/%d] %s",
            frame_idx - cfg.start_ind + 1,
            effective_end - cfg.start_ind,
            filename,
        )
        _process_file(frame_idx, filename, cfg, output_dir)

        if cfg.debug:
            log.info("Debug mode: stopping after first file.")
            break

    log.info(
        "Done. %d frame(s) → %s",
        effective_end - cfg.start_ind,
        output_dir,
    )
