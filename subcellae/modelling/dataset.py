"""
dataset.py
==========
PyTorch Dataset classes for .tif patch files.

The primary class is :class:`PatchDataset`, which always returns a 5-tuple::

    (image, condition, annotation_label, annotation_label_2, path)

where:

* ``image``              – (1, H, W) float32 tensor, values in [0, 1]
* ``condition``          – integer condition ID for the whole directory
                           (e.g. 0 = control, 1 = ycomp).  Used to distinguish
                           experimental groups; ignored by plain AE/VAE training.
* ``annotation_label``   – per-patch integer class from the primary annotation
                           file (e.g. FA type: 0–4), or ``-1`` if unlabelled.
                           Used by :func:`semisup_ae_loss`.
* ``annotation_label_2`` – per-patch integer class from an optional second
                           annotation file (e.g. Position: 0–4), or ``-1``.
                           Used by :func:`semisup_ae_loss_dual`.
* ``path``               – absolute path to the .tif file (str).

:class:`TIFFDataset` is kept as a backward-compatible wrapper that returns the
old 3-tuple ``(image, condition, path)`` (``label`` is now called
``condition`` internally but the argument name is unchanged).
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
import torch
from scipy.ndimage import rotate as _ndrotate
from torch.utils.data import DataLoader, Dataset, random_split

# ---------------------------------------------------------------------------
# Filename normalisation helper
# ---------------------------------------------------------------------------
# Patch files produced by the pipeline use underscore before the coordinate
# block: control_f0001x0112y0496ps32.tif
# Annotation CSVs (unique_ID column) use a hyphen: control-f0001x0112y0496ps32.tif
# This regex converts the patch filename to the annotation-CSV style so that
# the per-patch label lookup succeeds.
_COORD_UNDERSCORE = re.compile(r'_(f\d+x\d+y\d+ps\d+\.tiff?)$', re.IGNORECASE)

# Full coordinate-block parser: condition_str, frame_idx, cx_pad, cy_pad, ps
_COORD_RE = re.compile(
    r'^(.+)_f(\d+)x(\d+)y(\d+)ps(\d+)\.(tiff?)$', re.IGNORECASE
)


def _patch_name_to_annotation_key(filename: str) -> str:
    """Convert a patch filename to the annotation-CSV key style.

    ``control_f0001x0112y0496ps32.tif`` → ``control-f0001x0112y0496ps32.tif``
    """
    return _COORD_UNDERSCORE.sub(r'-\1', Path(filename).name)


# ---------------------------------------------------------------------------
# Primary unified class
# ---------------------------------------------------------------------------

class PatchDataset(Dataset):
    """Unified dataset for .tif image patches.

    Always returns ``(image, condition, annotation_label, path)``.

    Parameters
    ----------
    root_dir : str
        Directory containing .tif patch files.
    condition : int
        Integer ID for the experimental condition of this directory
        (e.g. ``0`` = control, ``1`` = ycomp).  Applied uniformly to every
        patch in the directory.
    condition_name : str
        Human-readable name for the condition (used in log output only).
    annotation_file : str or None
        Path to a CSV or Excel file with per-patch annotation labels.
        If ``None`` (default), ``annotation_label`` is ``-1`` for every patch.
    label_col : str
        Column in the annotation file to use as the class label
        (e.g. ``"Classification"`` or ``"position"``).
    filename_col : str
        Column in the annotation file that holds patch basenames
        (e.g. ``"crop_img_filename"`` or ``"unique_ID"``).
    label_order : list[str] or None
        Ordered list of string labels that defines the integer mapping
        (index 0 → class 0, …).  If ``None``, unique values are sorted
        alphabetically.
    transform : callable or None
        Applied to the raw ``(H, W)`` float32 numpy array before it is
        wrapped in a tensor.
    """

    def __init__(
        self,
        root_dir: str,
        condition: int = 0,
        condition_name: str = "",
        annotation_file: str | None = None,
        label_col: str = "Classification",
        filename_col: str = "crop_img_filename",
        label_order: list | None = None,
        annotation_file_2: str | None = None,
        label_col_2: str = "Position",
        filename_col_2: str = "crop_img_filename",
        label_order_2: list | None = None,
        transform=None,
        hist_map: "np.ndarray | None" = None,
    ):
        self.root_dir       = root_dir
        self.condition      = condition
        self.condition_name = condition_name or str(condition)
        self.transform      = transform
        # hist_map: (2, N) array — row 0 = src_q, row 1 = ref_q
        # applied as np.interp(image, src_q, ref_q) at load time
        self._hist_map = hist_map

        # ---- helper: load one annotation file → {filename: int} ----
        def _load_annotations(ann_file, col, fname_col, order):
            if not ann_file:
                return {}, order or [], {}, 0
            ann_path = Path(ann_file)
            ann_df   = (pd.read_excel(ann_path)
                        if ann_path.suffix.lower() in {".xlsx", ".xls"}
                        else pd.read_csv(ann_path))
            ann_df[fname_col] = ann_df[fname_col].astype(str).apply(lambda p: Path(p).name)
            fname_to_str = dict(zip(ann_df[fname_col], ann_df[col].astype(str)))
            if not order:
                order = sorted({v for v in fname_to_str.values() if v and v != "nan"})
            lbl_to_int = {lbl: i for i, lbl in enumerate(order)}
            fname_to_int = {f: lbl_to_int.get(s, -1) for f, s in fname_to_str.items()}
            return fname_to_int, order, lbl_to_int, len(order)

        # ---- primary annotation ----
        fname_to_ann1, self.label_order, self.label_to_int, self.num_classes = \
            _load_annotations(annotation_file, label_col, filename_col, label_order or [])

        # ---- secondary annotation ----
        fname_to_ann2, self.label_order_2, self.label_to_int_2, self.num_classes_2 = \
            _load_annotations(annotation_file_2, label_col_2, filename_col_2, label_order_2 or [])

        # ---- load images ----
        all_paths = sorted([
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.lower().endswith(("tif", "tiff"))
        ])

        self.data                = []
        self.paths               = []
        self.annotation_labels   = []
        self.annotation_labels_2 = []

        for img_path in all_paths:
            try:
                image = tiff.imread(img_path).astype(np.float32)
                if self._hist_map is not None:
                    src_q, ref_q = self._hist_map[0], self._hist_map[1]
                    image = np.interp(image, src_q, ref_q).astype(np.float32)
                if self.transform:
                    image = self.transform(image)
                image = torch.tensor(image, dtype=torch.float32)
            except Exception as e:
                print(f"Warning: Skipping unreadable image {img_path} – {e}")
                continue

            key = _patch_name_to_annotation_key(img_path)
            self.data.append(image)
            self.paths.append(img_path)
            self.annotation_labels.append(fname_to_ann1.get(key, -1))
            self.annotation_labels_2.append(fname_to_ann2.get(key, -1))

        n_ann  = sum(1 for l in self.annotation_labels   if l >= 0)
        n_ann2 = sum(1 for l in self.annotation_labels_2 if l >= 0)
        ann_info = ""
        if annotation_file:
            ann_info += f"  label1={label_col}: {n_ann} annotated"
        if annotation_file_2:
            ann_info += f"  label2={label_col_2}: {n_ann2} annotated"
        print(
            f"PatchDataset [{self.condition_name}]: {len(self.data)} patches, "
            f"condition={condition},{ann_info or ' (unlabelled)'}"
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return (
            self.data[idx],
            self.condition,
            self.annotation_labels[idx],
            self.annotation_labels_2[idx],
            self.paths[idx],
        )


# ---------------------------------------------------------------------------
# Multi-channel dataset
# ---------------------------------------------------------------------------

class MultiChannelPatchDataset(Dataset):
    """Dataset that stacks single-channel patches from multiple directories.

    Patches are matched by filename across all ``channel_dirs``.  Only filenames
    present in **every** directory are included (intersection).  Matched patches
    are stacked into a ``(C, H, W)`` tensor where ``C = len(channel_dirs)``.

    Returns the same 5-tuple as :class:`PatchDataset`::

        (image, condition, annotation_label, annotation_label_2, path)

    where ``image`` is ``(C, H, W)`` float32 and ``path`` is the first channel's
    file path.  Annotation lookup uses the first channel's filename as the key.

    Parameters
    ----------
    channel_dirs : list[str]
        Ordered list of directories, one per channel (e.g.
        ``[".../tiff_patches32_ch0", ".../tiff_patches32_ch1", ...]``).
        Must contain at least 2 entries.
    condition : int
        Integer condition ID applied uniformly to every patch.
    condition_name : str
        Human-readable condition label (for logging only).
    annotation_file : str or None
        Path to primary annotation CSV/Excel. ``None`` → all labels ``-1``.
    label_col : str
        Column used as primary class label.
    filename_col : str
        Column holding patch filenames in the annotation file.
    label_order : list[str] or None
        Ordered label list for integer mapping. ``None`` → alphabetical sort.
    annotation_file_2 : str or None
        Optional secondary annotation file.
    label_col_2, filename_col_2, label_order_2 :
        Same as above for the secondary annotation.
    transform : callable or None
        Applied to each raw ``(H, W)`` float32 numpy array before stacking.
    """

    def __init__(
        self,
        channel_dirs: list,
        condition: int = 0,
        condition_name: str = "",
        annotation_file: str | None = None,
        label_col: str = "Classification",
        filename_col: str = "crop_img_filename",
        label_order: list | None = None,
        annotation_file_2: str | None = None,
        label_col_2: str = "Position",
        filename_col_2: str = "crop_img_filename",
        label_order_2: list | None = None,
        transform=None,
    ):
        if len(channel_dirs) < 2:
            raise ValueError(
                f"MultiChannelPatchDataset requires at least 2 channel directories, "
                f"got {len(channel_dirs)}."
            )

        self.channel_dirs   = channel_dirs
        self.condition      = condition
        self.condition_name = condition_name or str(condition)
        self.transform      = transform

        # ---- annotation loader (identical logic to PatchDataset) ----
        def _load_annotations(ann_file, col, fname_col, order):
            if not ann_file:
                return {}, order or [], {}, 0
            ann_path = Path(ann_file)
            ann_df   = (pd.read_excel(ann_path)
                        if ann_path.suffix.lower() in {".xlsx", ".xls"}
                        else pd.read_csv(ann_path))
            ann_df[fname_col] = ann_df[fname_col].astype(str).apply(lambda p: Path(p).name)
            fname_to_str = dict(zip(ann_df[fname_col], ann_df[col].astype(str)))
            if not order:
                order = sorted({v for v in fname_to_str.values() if v and v != "nan"})
            lbl_to_int   = {lbl: i for i, lbl in enumerate(order)}
            fname_to_int = {f: lbl_to_int.get(s, -1) for f, s in fname_to_str.items()}
            return fname_to_int, order, lbl_to_int, len(order)

        fname_to_ann1, self.label_order, self.label_to_int, self.num_classes = \
            _load_annotations(annotation_file, label_col, filename_col, label_order or [])
        fname_to_ann2, self.label_order_2, self.label_to_int_2, self.num_classes_2 = \
            _load_annotations(annotation_file_2, label_col_2, filename_col_2, label_order_2 or [])

        # ---- find filenames present in every channel directory ----
        def _tif_names(d):
            return {f for f in os.listdir(d) if f.lower().endswith(("tif", "tiff"))}

        common = _tif_names(channel_dirs[0])
        for d in channel_dirs[1:]:
            common &= _tif_names(d)
        common = sorted(common)

        n_missing = sum(
            len(_tif_names(d)) for d in channel_dirs
        ) // len(channel_dirs) - len(common)
        if n_missing > 0:
            print(
                f"MultiChannelPatchDataset: {n_missing} patch(es) dropped "
                f"(not present in all {len(channel_dirs)} channel directories)."
            )

        # ---- load and stack ----
        self.data                = []
        self.paths               = []
        self.annotation_labels   = []
        self.annotation_labels_2 = []

        for fname in common:
            try:
                planes = []
                for d in channel_dirs:
                    ch_arr = tiff.imread(os.path.join(d, fname)).astype(np.float32)
                    if self.transform:
                        ch_arr = self.transform(ch_arr)
                    planes.append(ch_arr)
                # stack → (C, H, W)
                image = torch.tensor(np.stack(planes, axis=0), dtype=torch.float32)
            except Exception as e:
                print(f"Warning: Skipping {fname} – {e}")
                continue

            key = _patch_name_to_annotation_key(fname)
            self.data.append(image)
            self.paths.append(os.path.join(channel_dirs[0], fname))
            self.annotation_labels.append(fname_to_ann1.get(key, -1))
            self.annotation_labels_2.append(fname_to_ann2.get(key, -1))

        n_ann  = sum(1 for l in self.annotation_labels   if l >= 0)
        n_ann2 = sum(1 for l in self.annotation_labels_2 if l >= 0)
        ann_info = ""
        if annotation_file:
            ann_info += f"  label1={label_col}: {n_ann} annotated"
        if annotation_file_2:
            ann_info += f"  label2={label_col_2}: {n_ann2} annotated"
        print(
            f"MultiChannelPatchDataset [{self.condition_name}]: {len(self.data)} patches, "
            f"{len(channel_dirs)} channels, condition={condition}"
            f"{ann_info or ', (unlabelled)'}"
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return (
            self.data[idx],
            self.condition,
            self.annotation_labels[idx],
            self.annotation_labels_2[idx],
            self.paths[idx],
        )


# ---------------------------------------------------------------------------
# On-the-fly jitter + rotation crop dataset
# ---------------------------------------------------------------------------

class JitterCropDataset(Dataset):
    """On-the-fly translation jitter + rotation crop from source frame TIFFs.

    At init, loads full-frame CIO-RB normalized TIFFs from ``frame_dir`` into
    memory (one array per unique frame).  At each ``__getitem__`` a random
    translation jitter (dx, dy) and rotation angle are sampled, a context
    region is extracted from the source frame, rotated, then center-cropped to
    ``patch_size × patch_size``.

    Patch filenames must follow the patchprep convention::

        {condition_str}_f{frame_idx:04d}x{cx_padded:04d}y{cy_padded:04d}ps{ps}.tif

    where cx_padded / cy_padded encode coordinates that include the patchprep
    padding offset (``pad_size``; typically 64).  The dataset subtracts
    ``pad_size`` to recover the unpadded frame coordinates used in source TIFFs.

    Returns the same 5-tuple as :class:`PatchDataset`::

        (image, condition, annotation_label, annotation_label_2, path)

    where ``image`` is ``(1, patch_size, patch_size)`` float32 tensor.
    """

    def __init__(
        self,
        patch_dir: str,
        frame_dir: str,
        channel: str,
        condition: int = 0,
        condition_name: str = "",
        max_shift_px: int = 4,
        max_angle_deg: float = 15.0,
        patch_size: int = 32,
        pad_size: int = 64,
        annotation_file: str | None = None,
        label_col: str = "Classification",
        filename_col: str = "crop_img_filename",
        label_order: list | None = None,
        annotation_file_2: str | None = None,
        label_col_2: str = "Position",
        filename_col_2: str = "crop_img_filename",
        label_order_2: list | None = None,
    ):
        self.condition      = condition
        self.condition_name = condition_name or str(condition)
        self.max_shift_px   = max_shift_px
        self.max_angle_deg  = max_angle_deg
        self.patch_size     = patch_size

        # Minimum context size to guarantee a clean crop after max rotation.
        # Derivation: rotate a ps×ps square by θ → bounding box = ps*(|cosθ|+|sinθ|).
        θ = math.radians(max_angle_deg)
        ctx = math.ceil(patch_size * (math.cos(θ) + math.sin(θ))) + 4
        self._ctx = ctx + (ctx % 2)  # round up to even

        # ── annotation loader ───────────────────────────────────────────
        def _load_annotations(ann_file, col, fname_col, order):
            if not ann_file:
                return {}, order or [], {}, 0
            ann_path = Path(ann_file)
            ann_df   = (pd.read_excel(ann_path)
                        if ann_path.suffix.lower() in {".xlsx", ".xls"}
                        else pd.read_csv(ann_path))
            ann_df[fname_col] = ann_df[fname_col].astype(str).apply(lambda p: Path(p).name)
            fname_to_str = dict(zip(ann_df[fname_col], ann_df[col].astype(str)))
            if not order:
                order = sorted({v for v in fname_to_str.values() if v and v != "nan"})
            lbl_to_int   = {lbl: i for i, lbl in enumerate(order)}
            fname_to_int = {f: lbl_to_int.get(s, -1) for f, s in fname_to_str.items()}
            return fname_to_int, order, lbl_to_int, len(order)

        fname_to_ann1, self.label_order, self.label_to_int, self.num_classes = \
            _load_annotations(annotation_file, label_col, filename_col, label_order or [])
        fname_to_ann2, self.label_order_2, self.label_to_int_2, self.num_classes_2 = \
            _load_annotations(annotation_file_2, label_col_2, filename_col_2, label_order_2 or [])

        # ── parse patch filenames, load frames ───────────────────────────
        patch_dir = Path(patch_dir)
        frame_dir = Path(frame_dir)

        # Frame cache: (condition_str, frame_idx) → np.ndarray (H, W) float32
        _frame_cache: dict = {}

        def _get_frame(cond_str, fidx):
            key = (cond_str, fidx)
            if key not in _frame_cache:
                tif_path = frame_dir / f"{cond_str}_f{fidx:04d}_{channel}.tif"
                _frame_cache[key] = tiff.imread(str(tif_path)).astype(np.float32)
            return _frame_cache[key]

        # entries[i] = (frame_key, cx_frame, cy_frame, path)
        # frame_key: (cond_str, fidx) for self._frames lookup
        self._frames             = {}
        self._entry_keys         = []   # (frame_key, cx_frame, cy_frame)
        self.paths               = []
        self.annotation_labels   = []
        self.annotation_labels_2 = []

        for fname in sorted(os.listdir(str(patch_dir))):
            if not fname.lower().endswith(("tif", "tiff")):
                continue
            m = _COORD_RE.match(fname)
            if not m:
                continue
            cond_str = m.group(1)
            fidx     = int(m.group(2))
            cx_pad   = int(m.group(3))
            cy_pad   = int(m.group(4))

            cx_frame = cx_pad - pad_size
            cy_frame = cy_pad - pad_size

            fkey = (cond_str, fidx)
            if fkey not in self._frames:
                try:
                    self._frames[fkey] = _get_frame(cond_str, fidx)
                except Exception as e:
                    print(f"Warning: Cannot load frame {cond_str}_f{fidx:04d}_{channel}.tif – {e}")
                    continue

            full_path = str(patch_dir / fname)
            key = _patch_name_to_annotation_key(fname)
            self._entry_keys.append((fkey, cx_frame, cy_frame))
            self.paths.append(full_path)
            self.annotation_labels.append(fname_to_ann1.get(key, -1))
            self.annotation_labels_2.append(fname_to_ann2.get(key, -1))

        # Pre-pad all frames once at init so __getitem__ only does a slice + rotate.
        # pad = h2 + max_shift + 2 must match the offset applied to cx/cy below.
        self._pad_px = self._ctx // 2 + max_shift_px + 2
        self._frames_padded = {
            fkey: np.pad(arr, self._pad_px, mode='reflect')
            for fkey, arr in self._frames.items()
        }

        n_ann  = sum(1 for l in self.annotation_labels   if l >= 0)
        n_ann2 = sum(1 for l in self.annotation_labels_2 if l >= 0)
        ann_info = ""
        if annotation_file:
            ann_info += f"  label1={label_col}: {n_ann} annotated"
        if annotation_file_2:
            ann_info += f"  label2={label_col_2}: {n_ann2} annotated"
        print(
            f"JitterCropDataset [{self.condition_name}]: {len(self._entry_keys)} patches, "
            f"{len(self._frames)} frames, condition={condition}, ch={channel}, "
            f"ctx={self._ctx}px, shift±{max_shift_px}px, rot±{max_angle_deg}°"
            f"{ann_info or ', (unlabelled)'}"
        )

    def __len__(self) -> int:
        return len(self._entry_keys)

    def __getitem__(self, idx):
        fkey, cx0, cy0 = self._entry_keys[idx]
        frame_p = self._frames_padded[fkey]

        dx    = int(np.random.randint(-self.max_shift_px, self.max_shift_px + 1))
        dy    = int(np.random.randint(-self.max_shift_px, self.max_shift_px + 1))
        angle = float(np.random.uniform(-self.max_angle_deg, self.max_angle_deg))

        h2   = self._ctx // 2
        cx_p = cx0 + dx + self._pad_px
        cy_p = cy0 + dy + self._pad_px

        ctx_patch = frame_p[cy_p - h2 : cy_p + h2, cx_p - h2 : cx_p + h2]

        if abs(angle) > 0.01:
            ctx_patch = _ndrotate(ctx_patch, angle, reshape=False, mode='reflect', order=1)

        ps     = self.patch_size
        offset = (self._ctx - ps) // 2
        patch  = ctx_patch[offset : offset + ps, offset : offset + ps]

        image = torch.tensor(patch[np.newaxis], dtype=torch.float32)
        return (
            image,
            self.condition,
            self.annotation_labels[idx],
            self.annotation_labels_2[idx],
            self.paths[idx],
        )


# ---------------------------------------------------------------------------
# Enlarged-crop dataset (no augmentation at load time — for contrastive AE)
# ---------------------------------------------------------------------------

class EnlargedCropDataset(Dataset):
    """Context-enlarged crop dataset for contrastive augmentation.

    Loads full-frame CIO-RB normalized TIFFs at init. At each ``__getitem__``,
    returns a ``(1, context_size, context_size)`` float32 tensor centered on the
    FA patch position with **no rotation or translation**.

    The oversized context lets the training loop apply independent random
    ±shift translation + ±angle rotation for both contrastive views via a
    single GPU bilinear affine interpolation (affine_grid + grid_sample),
    avoiding double-interpolation artifacts.

    Safe context size formula::

        context_size = 2 * ceil(sqrt(2) * (patch_size/2 + max_shift_px))

    For defaults ps=32, max_shift=4:
        2 * ceil(sqrt(2) * 20) = 2 * ceil(28.28) = 2 * 29 = 58

    Returns the same 5-tuple as :class:`PatchDataset`::

        (image, condition, annotation_label, annotation_label_2, path)

    where ``image`` is ``(1, context_size, context_size)`` float32 tensor.
    """

    def __init__(
        self,
        patch_dir: str,
        frame_dir: str,
        channel: str,
        condition: int = 0,
        condition_name: str = "",
        context_size: int = 58,
        patch_size: int = 32,
        pad_size: int = 64,
        input_divisor: float = 1.0,
        input_clip_max: float | None = None,
        annotation_file: str | None = None,
        label_col: str = "Classification",
        filename_col: str = "crop_img_filename",
        label_order: list | None = None,
        annotation_file_2: str | None = None,
        label_col_2: str = "Position",
        filename_col_2: str = "crop_img_filename",
        label_order_2: list | None = None,
    ):
        self.condition      = condition
        self.condition_name = condition_name or str(condition)
        self.context_size   = context_size + (context_size % 2)  # ensure even
        self._input_divisor = float(input_divisor)
        self._input_clip_max = float(input_clip_max) if input_clip_max is not None else None

        # ── annotation loader ────────────────────────────────────────────
        def _load_annotations(ann_file, col, fname_col, order):
            if not ann_file:
                return {}, order or [], {}, 0
            ann_path = Path(ann_file)
            ann_df   = (pd.read_excel(ann_path)
                        if ann_path.suffix.lower() in {".xlsx", ".xls"}
                        else pd.read_csv(ann_path))
            ann_df[fname_col] = ann_df[fname_col].astype(str).apply(lambda p: Path(p).name)
            fname_to_str = dict(zip(ann_df[fname_col], ann_df[col].astype(str)))
            if not order:
                order = sorted({v for v in fname_to_str.values() if v and v != "nan"})
            lbl_to_int   = {lbl: i for i, lbl in enumerate(order)}
            fname_to_int = {f: lbl_to_int.get(s, -1) for f, s in fname_to_str.items()}
            return fname_to_int, order, lbl_to_int, len(order)

        fname_to_ann1, self.label_order, self.label_to_int, self.num_classes = \
            _load_annotations(annotation_file, label_col, filename_col, label_order or [])
        fname_to_ann2, self.label_order_2, self.label_to_int_2, self.num_classes_2 = \
            _load_annotations(annotation_file_2, label_col_2, filename_col_2, label_order_2 or [])

        # ── parse patch filenames, load frames ───────────────────────────
        patch_dir = Path(patch_dir)
        frame_dir = Path(frame_dir)

        _frame_cache: dict = {}

        def _get_frame(cond_str, fidx):
            key = (cond_str, fidx)
            if key not in _frame_cache:
                tif_path = frame_dir / f"{cond_str}_f{fidx:04d}_{channel}.tif"
                _frame_cache[key] = tiff.imread(str(tif_path)).astype(np.float32)
            return _frame_cache[key]

        self._frames             = {}
        self._entry_keys         = []   # (frame_key, cx_frame, cy_frame)
        self.paths               = []
        self.annotation_labels   = []
        self.annotation_labels_2 = []

        for fname in sorted(os.listdir(str(patch_dir))):
            if not fname.lower().endswith(("tif", "tiff")):
                continue
            m = _COORD_RE.match(fname)
            if not m:
                continue
            cond_str = m.group(1)
            fidx     = int(m.group(2))
            cx_pad   = int(m.group(3))
            cy_pad   = int(m.group(4))

            cx_frame = cx_pad - pad_size
            cy_frame = cy_pad - pad_size

            fkey = (cond_str, fidx)
            if fkey not in self._frames:
                try:
                    self._frames[fkey] = _get_frame(cond_str, fidx)
                except Exception as e:
                    print(f"Warning: Cannot load frame {cond_str}_f{fidx:04d}_{channel}.tif – {e}")
                    continue

            full_path = str(patch_dir / fname)
            key = _patch_name_to_annotation_key(fname)
            self._entry_keys.append((fkey, cx_frame, cy_frame))
            self.paths.append(full_path)
            self.annotation_labels.append(fname_to_ann1.get(key, -1))
            self.annotation_labels_2.append(fname_to_ann2.get(key, -1))

        # Pre-pad all frames once at init so __getitem__ is a simple slice.
        # pad = h2 + 4 must match the offset applied to cx_frame/cy_frame below.
        self._pad_px = self.context_size // 2 + 4
        self._frames_padded = {
            fkey: np.pad(arr, self._pad_px, mode='reflect')
            for fkey, arr in self._frames.items()
        }

        n_ann  = sum(1 for l in self.annotation_labels   if l >= 0)
        n_ann2 = sum(1 for l in self.annotation_labels_2 if l >= 0)
        ann_info = ""
        if annotation_file:
            ann_info += f"  label1={label_col}: {n_ann} annotated"
        if annotation_file_2:
            ann_info += f"  label2={label_col_2}: {n_ann2} annotated"
        print(
            f"EnlargedCropDataset [{self.condition_name}]: {len(self._entry_keys)} patches, "
            f"{len(self._frames)} frames, condition={condition}, ch={channel}, "
            f"ctx={self.context_size}px (ps={patch_size})"
            f"{ann_info or ', (unlabelled)'}"
        )

    def __len__(self) -> int:
        return len(self._entry_keys)

    def __getitem__(self, idx):
        fkey, cx0, cy0 = self._entry_keys[idx]
        frame_p = self._frames_padded[fkey]

        h2   = self.context_size // 2
        cx_p = cx0 + self._pad_px
        cy_p = cy0 + self._pad_px

        region = frame_p[cy_p - h2 : cy_p + h2, cx_p - h2 : cx_p + h2]
        if self._input_clip_max is not None:
            region = np.clip(region, 0.0, self._input_clip_max)
        if self._input_divisor != 1.0:
            region = region / self._input_divisor
        image  = torch.tensor(region[np.newaxis], dtype=torch.float32)
        return (
            image,
            self.condition,
            self.annotation_labels[idx],
            self.annotation_labels_2[idx],
            self.paths[idx],
        )


# ---------------------------------------------------------------------------
# Multi-channel enlarged-crop dataset
# ---------------------------------------------------------------------------

class MultiChannelEnlargedCropDataset(Dataset):
    """Enlarged-crop dataset for multi-channel (stacked) contrastive AE.

    Loads full-frame TIFFs for each channel at init.  At each ``__getitem__``,
    extracts the same (context_size × context_size) region from every channel
    and stacks them into a ``(C, context_size, context_size)`` float32 tensor.

    Coordinates are parsed from ``patch_dirs[0]``; all channels must have
    matching filenames (same FA positions, different channel signal).

    Parameters
    ----------
    patch_dirs : list[str]
        One directory per channel (e.g. ch1/pax dir, ch3/act dir).
    frame_dir : str
        Directory containing per-frame TIFFs named
        ``{cond}_f{idx:04d}_{channel}.tif``.
    channels : list[str]
        Channel suffixes matching ``patch_dirs`` order (e.g. ``["pax", "act"]``).
    """

    def __init__(
        self,
        patch_dirs: list,
        frame_dir: str,
        channels: list,
        condition: int = 0,
        condition_name: str = "",
        context_size: int = 58,
        patch_size: int = 32,
        pad_size: int = 64,
        input_divisor: float = 1.0,
        input_clip_max: float | None = None,
        annotation_file: str | None = None,
        label_col: str = "Classification",
        filename_col: str = "crop_img_filename",
        label_order: list | None = None,
        annotation_file_2: str | None = None,
        label_col_2: str = "Position",
        filename_col_2: str = "crop_img_filename",
        label_order_2: list | None = None,
    ):
        if len(patch_dirs) != len(channels):
            raise ValueError(
                f"patch_dirs and channels must have the same length, "
                f"got {len(patch_dirs)} and {len(channels)}."
            )
        self.condition      = condition
        self.condition_name = condition_name or str(condition)
        self.context_size   = context_size + (context_size % 2)
        self._input_divisor = float(input_divisor)
        self._input_clip_max = float(input_clip_max) if input_clip_max is not None else None
        self._n_channels    = len(channels)

        # ── annotation loader (same as EnlargedCropDataset) ──────────────
        def _load_annotations(ann_file, col, fname_col, order):
            if not ann_file:
                return {}, order or [], {}, 0
            ann_path = Path(ann_file)
            ann_df   = (pd.read_excel(ann_path)
                        if ann_path.suffix.lower() in {".xlsx", ".xls"}
                        else pd.read_csv(ann_path))
            ann_df[fname_col] = ann_df[fname_col].astype(str).apply(lambda p: Path(p).name)
            fname_to_str = dict(zip(ann_df[fname_col], ann_df[col].astype(str)))
            if not order:
                order = sorted({v for v in fname_to_str.values() if v and v != "nan"})
            lbl_to_int   = {lbl: i for i, lbl in enumerate(order)}
            fname_to_int = {f: lbl_to_int.get(s, -1) for f, s in fname_to_str.items()}
            return fname_to_int, order, lbl_to_int, len(order)

        fname_to_ann1, self.label_order, self.label_to_int, self.num_classes = \
            _load_annotations(annotation_file, label_col, filename_col, label_order or [])
        fname_to_ann2, self.label_order_2, self.label_to_int_2, self.num_classes_2 = \
            _load_annotations(annotation_file_2, label_col_2, filename_col_2, label_order_2 or [])

        # ── parse coordinates from first channel's patch dir ─────────────
        primary_dir = Path(patch_dirs[0])
        frame_dir   = Path(frame_dir)
        pad_px      = self.context_size // 2 + 4

        # per-channel frame cache: channel_idx → {(cond_str, fidx): padded_frame}
        self._frames_per_ch: list[dict] = [{} for _ in channels]

        self._entry_keys         = []   # (frame_key, cx_frame, cy_frame)
        self.paths               = []
        self.annotation_labels   = []
        self.annotation_labels_2 = []

        for fname in sorted(os.listdir(str(primary_dir))):
            if not fname.lower().endswith(("tif", "tiff")):
                continue
            m = _COORD_RE.match(fname)
            if not m:
                continue
            cond_str = m.group(1)
            fidx     = int(m.group(2))
            cx_pad   = int(m.group(3))
            cy_pad   = int(m.group(4))
            cx_frame = cx_pad - pad_size
            cy_frame = cy_pad - pad_size

            fkey = (cond_str, fidx)
            all_loaded = True
            for ci, ch in enumerate(channels):
                if fkey not in self._frames_per_ch[ci]:
                    tif_path = frame_dir / f"{cond_str}_f{fidx:04d}_{ch}.tif"
                    try:
                        arr = tiff.imread(str(tif_path)).astype(np.float32)
                        self._frames_per_ch[ci][fkey] = np.pad(arr, pad_px, mode='reflect')
                    except Exception as e:
                        print(f"Warning: Cannot load {tif_path.name} – {e}")
                        all_loaded = False
                        break
            if not all_loaded:
                continue

            key = _patch_name_to_annotation_key(fname)
            self._entry_keys.append((fkey, cx_frame, cy_frame, pad_px))
            self.paths.append(str(primary_dir / fname))
            self.annotation_labels.append(fname_to_ann1.get(key, -1))
            self.annotation_labels_2.append(fname_to_ann2.get(key, -1))

        n_ann  = sum(1 for l in self.annotation_labels   if l >= 0)
        n_ann2 = sum(1 for l in self.annotation_labels_2 if l >= 0)
        print(
            f"MultiChannelEnlargedCropDataset [{self.condition_name}]: "
            f"{len(self._entry_keys)} patches, channels={channels}, "
            f"condition={condition}, ctx={self.context_size}px"
            f"{f'  label1: {n_ann} annotated' if annotation_file else ', (unlabelled)'}"
        )

    def __len__(self) -> int:
        return len(self._entry_keys)

    def __getitem__(self, idx):
        fkey, cx0, cy0, pad_px = self._entry_keys[idx]
        h2   = self.context_size // 2
        cx_p = cx0 + pad_px
        cy_p = cy0 + pad_px

        planes = []
        for ch_frames in self._frames_per_ch:
            region = ch_frames[fkey][cy_p - h2 : cy_p + h2, cx_p - h2 : cx_p + h2]
            if self._input_clip_max is not None:
                region = np.clip(region, 0.0, self._input_clip_max)
            if self._input_divisor != 1.0:
                region = region / self._input_divisor
            planes.append(region)

        image = torch.tensor(np.stack(planes, axis=0), dtype=torch.float32)
        return (
            image,
            self.condition,
            self.annotation_labels[idx],
            self.annotation_labels_2[idx],
            self.paths[idx],
        )


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------

class TIFFDataset(Dataset):
    """Backward-compatible dataset returning ``(image, condition, path)``.

    Wraps :class:`PatchDataset`.  The argument ``label`` is accepted for
    historical reasons and maps to ``condition``.
    """

    def __init__(self, root_dir, label: int = 0, transform=None):
        self._ds = PatchDataset(root_dir, condition=label, transform=transform)
        # expose paths so grouped-split helpers can access them
        self.paths = self._ds.paths

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        image, condition, _, _, path = self._ds[idx]
        return image, condition, path


class AnnotatedTIFFDataset(Dataset):
    """Backward-compatible annotated dataset returning ``(image, annotation_label, path)``.

    Wraps :class:`PatchDataset`.
    """

    def __init__(
        self,
        root_dir,
        annotation_file,
        label_col,
        filename_col="crop_img_filename",
        label_order=None,
        transform=None,
    ):
        self._ds = PatchDataset(
            root_dir,
            condition=0,
            annotation_file=annotation_file,
            label_col=label_col,
            filename_col=filename_col,
            label_order=label_order,
            transform=transform,
        )
        self.label_order  = self._ds.label_order
        self.label_to_int = self._ds.label_to_int
        self.num_classes  = self._ds.num_classes
        self.paths        = self._ds.paths

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        image, _, annotation_label, _, path = self._ds[idx]
        return image, annotation_label, path
