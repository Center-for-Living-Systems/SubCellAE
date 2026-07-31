"""
ae_pipeline.py
==============
Training pipeline for the four autoencoder variants in subcellae.

Provides a single entry-point function `run_ae_pipeline()` that orchestrates:
  1. Config validation and logging
  2. Dataset construction from one or more patch directories
  3. Train / validation split
  4. DataLoader creation
  5. Model instantiation
  6. Training via the appropriate ``train_*`` function
  7. Final model checkpoint

Supported model types (set via ``AEConfig.model_type``):
  ``"ae"``          – standard convolutional autoencoder (AE)
  ``"vae"``         – variational AE / beta-VAE (VAE32)
  ``"semisup"``     – semi-supervised AE with classification head (SemiSupAE)
  ``"contrastive"`` – contrastive AE with NT-Xent loss (ContrastiveAE)
  ``"supcon"``      – supervised contrastive AE with SupCon loss (ContrastiveAE)

All heavy-lifting model code lives in subcellae/modelling/autoencoders.py; this
module only wires the pieces together.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")          # non-interactive backend – safe for scripts
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Subset, random_split

from subcellae.modelling.dataset import PatchDataset, MultiChannelPatchDataset, JitterCropDataset, EnlargedCropDataset, MultiChannelEnlargedCropDataset
from subcellae.modelling.autoencoders import (
    AE, train_ae,
    VAE32, train_vae,
    SemiSupAE, train_semisup_ae,
    ContrastiveAE, train_contrastive_ae,
    train_supervised_contrastive_ae,
)
from subcellae.modelling.intensity_transform import make_log_map

log = logging.getLogger(__name__)

_VALID_MODEL_TYPES = {"ae", "vae", "semisup", "contrastive", "supcon"}


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class AEConfig:
    """All tuneable parameters for one autoencoder training run.

    Parameters
    ----------
    result_dir : Path
        Directory where checkpoints, loss curves, and the final model are
        written.  Created automatically if it does not exist.
    patch_dirs : list[dict]
        Each entry must have ``path`` (str), ``condition`` (int, e.g. 0=control),
        and optionally ``condition_name`` (str).  Per-patch FA-type labels come
        from the annotation file, not from this field.
    model_type : str
        Which model to train: ``"ae"`` | ``"vae"`` | ``"semisup"`` |
        ``"contrastive"`` | ``"supcon"``.

    Model architecture
    ------------------
    latent_dim : int
        Bottleneck dimension.  Default ``8``.
    input_ps : int
        Spatial size of the (square) input patch.  Default ``32``.
    no_ch : int
        Number of input channels.  Default ``1``.
    BN_flag : bool
        Enable BatchNorm2d in conv layers.  Default ``False``.
    dropout_flag : bool
        Enable Dropout(0.3) in FC layers.  Default ``False``.

    VAE-specific
    ------------
    out_activation : str
        Output activation for VAE32: ``"sigmoid"`` or ``"identity"``.
    beta : float
        KL weight for the VAE ELBO (beta=1 → standard VAE).
    beta_anneal : bool
        Linearly warm up beta from 0 → beta over the first half of
        training (helps avoid posterior collapse).
    recon_type : str
        Reconstruction loss: ``"mse"`` or ``"bce"``.

    SemiSup-specific
    ----------------
    num_classes : int
        Number of target classes for the classification head.
    lambda_recon : float
        Weight on the reconstruction term.
    lambda_cls : float
        Weight on the classification term.

    Contrastive-specific
    --------------------
    proj_dim : int
        Output dimension of the NT-Xent projection head.
    noise_prob : float
        Salt-and-pepper corruption probability for the noisy view.
    temperature : float
        Temperature for the NT-Xent softmax.
    lambda_contrast : float
        Weight on the contrastive term.

    Training
    --------
    epochs : int
        Number of training epochs.  Default ``200``.
    lr : float
        Adam learning rate.  Default ``1e-3``.
    batch_size : int
        Mini-batch size.  Default ``128``.
    val_split : float
        Fraction of data held out for validation.  Default ``0.2``.
    loss_norm_flag : bool
        Use normalised MSE (signal-power-normalised) for the AE variant.
        Ignored by other model types.  Default ``False``.

    Device
    ------
    device : str
        ``"auto"`` selects CUDA if available, else CPU.
        Pass ``"cuda"`` or ``"cpu"`` to override.
    """

    # --- required ---
    result_dir: Path
    patch_dirs: list      # list of dicts: [{path, condition, condition_name}, ...]

    # --- model selection ---
    model_type: str = "ae"

    # --- shared architecture ---
    latent_dim: int   = 8
    input_ps: int     = 32
    no_ch: int        = 1
    BN_flag: bool     = False
    dropout_flag: bool = False

    # --- VAE-specific ---
    out_activation: str = "sigmoid"
    beta: float         = 1.0
    beta_anneal: bool   = False
    recon_type: str     = "mse"

    # --- SemiSup-specific ---
    num_classes: int    = 6
    lambda_recon: float = 1.0
    lambda_cls: float   = 1.0

    # --- annotation (SemiSup) ---
    annotation_file: str  = ""    # path to CSV/Excel with per-patch labels
    label_col: str        = "Classification"  # column to use as class label
    filename_col: str     = "crop_img_filename"  # column that holds patch basenames
    label_order: list     = None  # ordered list of string labels; None → auto alphabetical

    # --- second annotation (dual SemiSup, e.g. Position) ---
    annotation_file_2: str = ""      # same CSV is typical; leave "" to disable
    label_col_2: str       = "Position"
    filename_col_2: str    = "crop_img_filename"
    label_order_2: list    = None
    num_classes_2: int     = 0       # auto-set from label_order_2 if provided
    lambda_cls_2: float    = 0.0     # 0.0 = single-label mode (backward-compat)

    # --- Contrastive-specific ---
    proj_dim: int                  = 64
    noise_prob: float              = 0.05
    temperature: float             = 0.5
    lambda_contrast: float         = 0.5
    use_flip: bool                 = True         # random H/V flips in augmentation
    intensity_scale_range: tuple   = (0.8, 1.2)  # (low, high) intensity multiplier

    # --- training ---
    epochs: int         = 200
    lr: float           = 1e-3
    batch_size: int     = 128
    val_split: float    = 0.2
    loss_norm_flag: bool = False
    group_split: bool   = True   # keep all patches from the same image in the same split

    # --- LR scheduler ---
    lr_scheduler: str           = "none"   # "none" | "plateau" | "cosine"
    lr_scheduler_patience: int  = 20       # epochs without improvement before reducing LR (plateau)
    lr_scheduler_factor: float  = 0.5     # LR reduction factor (plateau)
    lr_min: float               = 1e-6    # minimum LR floor

    # --- regularisation / training stability ---
    weight_decay: float          = 1e-4   # Adam L2 weight decay (all model types)
    early_stopping_patience: int = 0      # 0 = disabled
    min_epochs_for_best: int     = 200    # best-checkpoint tracking starts at this epoch
    warmup_epochs: int           = 200    # recon-only phase before adding cls loss
    semisup_lr_scheduler_patience: int = 10  # semisup ReduceLROnPlateau patience; 0 = disabled

    # --- reconstruction output ---
    save_recon: bool       = True   # whether to write reconstruction images
    compact_recon: bool    = False  # sweep mode: skip raw TIFs, limit recon to 4 images
                                    # (1st train+val per condition); saves ~10× disk space
    recon_pad_size: int    = 64     # padding used during patch extraction; subtracted
                                    # to convert padded → original image coordinates
    recon_image_size: int  = 1024   # fixed canvas height/width (px) for whole-image tifs

    # --- device ---
    device: str = "auto"

    # --- data loading ---
    num_workers: int = 0   # DataLoader worker processes; set >0 when __getitem__ is CPU-heavy

    # --- histogram matching ---
    # Path to directory containing {dataset}_map.npz files produced by
    # scripts/compute_histogram_maps.py.  If set, each patch_dir will have
    # histogram matching applied automatically (dataset name auto-detected
    # from the path: .../cio_rb/<dataset>/...).
    hist_map_dir: str | None = None

    # --- jitter crop (on-the-fly augmentation from source frame TIFFs) ---
    # When enabled, each patch_dirs entry must supply a "frame_dir" key pointing
    # to the directory of per-channel source frame TIFFs produced by
    # run_frameextract_from_config.py  (e.g. ae_results/source_frames/cio_rb/vinc/control).
    jitter_crop: bool            = False
    jitter_crop_channel: str     = "pax"   # channel name used in frame TIFF filenames
    jitter_crop_max_shift: int   = 4       # max translation jitter (px each direction)
    jitter_crop_max_angle: float = 15.0    # max rotation (degrees)
    jitter_crop_pad_size: int    = 64      # patchprep padding offset in patch coordinates

    # --- enlarged crop (context-only crop; jitter+rotation done per-view in training loop) ---
    # Returns (1, context_size, context_size) patch with no augmentation at load time.
    # The training loop calls _jitter_rot_crop independently for each contrastive view,
    # enabling true per-view translation + rotation diversity in a single bilinear pass.
    # context_size formula: 2 * ceil(sqrt(2) * (input_ps/2 + max_shift_px))
    # For input_ps=32, max_shift=4: context_size=58.
    enlarged_crop: bool              = False
    enlarged_crop_channel: "str | list" = "pax"
    enlarged_crop_context_size: int  = 58
    enlarged_crop_max_shift: int     = 4
    enlarged_crop_max_angle: float   = 15.0
    enlarged_crop_pad_size: int      = 64
    enlarged_crop_input_divisor: float = 1.0
    enlarged_crop_input_clip_max: float | None = None

    output_sigmoid: bool             = True
    recon_loss_type: str             = "mse"   # "mse", "l1", "nmse", "nl1"
    lambda_hessian: float            = 0.0     # additive Hessian-L1 regulariser weight

    # --- patch input divisor (non-enlcrop path; mirrors enlarged_crop_input_divisor) ---
    patch_input_divisor: float = 1.0  # divide raw patch values by this factor at load time

    # --- shifted-log intensity mapping (applied inside training loop) ---
    # When enabled: x_ciorb → log_map_forward(x) → model → log_map_inverse → metrics
    log_map_enabled: bool  = False
    log_map_x_min: float   = -0.03   # lower bound → 0 in log space; covers global min -0.0296
    log_map_x_ref: float   = 10.0    # upper reference → 1 in log space; covers max ~10.7
    log_map_delta: float   = 0.5     # compression strength

    def __post_init__(self):
        # Coerce and create result_dir
        self.result_dir = Path(self.result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)

        # Validate model_type
        if self.model_type not in _VALID_MODEL_TYPES:
            raise ValueError(
                f"model_type must be one of {_VALID_MODEL_TYPES}, "
                f"got {self.model_type!r}"
            )

        # Resolve device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def _extract_group_key(path: str) -> str:
    """Return the image-level group key from a patch filename.

    Filename format: ``{prefix}_f{NNNN}x{xxxx}y{yyyy}ps{pp}.tif``
    Group key      : ``{dataset}_{prefix}_f{NNNN}``

    The dataset prefix is the directory two levels above the patch dir
    (e.g. .../vinc/control/tiff_patches32_mr10/fname → "vinc").
    This prevents frames from different datasets with the same filename
    (e.g. vinc control_f0001 vs ppax control_f0001) from being merged.
    Falls back to the full stem if the pattern is not matched.
    """
    stem = Path(path).stem   # strip .tif
    m = re.match(r'^(.+_f\d+)x\d+', stem)
    frame_group = m.group(1) if m else stem
    # dataset prefix: grandparent of the patch directory
    try:
        dataset_prefix = Path(path).parents[2].name
    except IndexError:
        dataset_prefix = ""
    return f"{dataset_prefix}_{frame_group}" if dataset_prefix else frame_group


def _grouped_train_val_split(
    datasets: list,
    val_split: float,
    per_ds_val_splits: list | None = None,
    seed: int = 42,
) -> tuple:
    """Split dataset indices so all patches from the same image stay together.

    Parameters
    ----------
    datasets : list
        Individual dataset objects (each must have a ``paths`` attribute).
    val_split : float
        Global fraction of *images* (groups) to hold out for validation.
        Used when ``per_ds_val_splits`` is None.
    per_ds_val_splits : list[float] | None
        Per-dataset val_split fractions (one per element of ``datasets``).
        When provided, each dataset is split independently with its own
        fraction, using a fresh RandomState(seed) for each dataset so
        repeated datasets (same paths) get identical splits.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    train_indices, val_indices : list[int], list[int]
        Flat indices into the concatenated dataset.
    """
    if per_ds_val_splits is None:
        # Original behaviour: pool all groups across all datasets, split globally.
        all_paths = []
        for ds in datasets:
            all_paths.extend(getattr(ds, "paths", []))

        group_to_indices: dict = defaultdict(list)
        for idx, path in enumerate(all_paths):
            group_to_indices[_extract_group_key(path)].append(idx)

        groups = sorted(group_to_indices.keys())
        rng = np.random.RandomState(seed)
        rng.shuffle(groups)

        n_val = max(1, int(len(groups) * val_split))
        val_groups   = set(groups[:n_val])
        train_groups = set(groups[n_val:])

        train_indices = [i for g in sorted(train_groups) for i in group_to_indices[g]]
        val_indices   = [i for g in sorted(val_groups)   for i in group_to_indices[g]]
        return train_indices, val_indices
    else:
        # Per-dataset split: each dataset is split independently.
        # Using a fresh RandomState(seed) per dataset ensures repeated datasets
        # (same paths, different repeat index) always produce identical splits.
        train_indices, val_indices = [], []
        offset = 0
        for ds, ds_val_split in zip(datasets, per_ds_val_splits):
            ds_paths = getattr(ds, "paths", [])
            group_to_local: dict = defaultdict(list)
            for local_idx, path in enumerate(ds_paths):
                group_to_local[_extract_group_key(path)].append(local_idx)

            groups = sorted(group_to_local.keys())
            rng = np.random.RandomState(seed)
            rng.shuffle(groups)

            n_val = max(1, int(len(groups) * ds_val_split))
            val_gs   = set(groups[:n_val])
            train_gs = set(groups[n_val:])

            train_indices.extend(offset + i for g in sorted(train_gs) for i in group_to_local[g])
            val_indices.extend(  offset + i for g in sorted(val_gs)   for i in group_to_local[g])
            offset += len(ds)

        return train_indices, val_indices


# ---------------------------------------------------------------------------
# Coordinate parsing
# ---------------------------------------------------------------------------

_COORD_RE = re.compile(r'^(.+_f\d+)x(\d+)y(\d+)ps(\d+)')

def _parse_patch_coords(filename: str, path: str | None = None):
    """Parse patch coordinates from a filename.

    Filename format: ``{group}_x{xxxx}y{yyyy}ps{pp}.tif``
    Example: ``control_f0001x0080y0816ps32.tif``

    Returns
    -------
    (group, x_c, y_c, ps) : (str, int, int, int)
        ``group`` is the source-image key (e.g. ``vinc_control_f0001``).
        ``x_c``, ``y_c`` are the patch centre coordinates in padded image space.
        ``ps`` is the patch side length in pixels.
    Returns ``None`` if the filename does not match the expected pattern.
    """
    stem = Path(filename).stem
    m = _COORD_RE.match(stem)
    if m is None:
        return None
    frame_group = m.group(1)
    if path is not None:
        try:
            dataset_prefix = Path(path).parents[2].name
            frame_group = f"{dataset_prefix}_{frame_group}"
        except IndexError:
            pass
    return frame_group, int(m.group(2)), int(m.group(3)), int(m.group(4))


# ---------------------------------------------------------------------------
# Latent extraction and CSV export
# ---------------------------------------------------------------------------

def _patch_hessian_l1(raw: np.ndarray, recon: np.ndarray) -> float:
    """Mean Frobenius norm of the Hessian of the residual (raw − recon).

    Computes second-order finite differences of the residual image at interior
    pixels, then returns mean(‖H_residual‖_F) over all interior pixels.
    This directly measures curvature content left in the reconstruction error,
    independent of the absolute curvature scale of the raw patch.
    For multi-channel (C, H, W) the result is averaged over channels.
    """
    if raw.ndim == 3:  # (C, H, W) — average over channels
        return float(np.mean([_patch_hessian_l1(raw[c], recon[c])
                               for c in range(raw.shape[0])]))
    # single channel: (H, W) — work on residual image
    d = raw.astype(np.float64) - recon.astype(np.float64)
    dIxx = d[1:-1, 2:]  + d[1:-1, :-2] - 2 * d[1:-1, 1:-1]
    dIyy = d[2:,  1:-1] + d[:-2, 1:-1] - 2 * d[1:-1, 1:-1]
    dIxy = (d[2:, 2:] - d[2:, :-2] - d[:-2, 2:] + d[:-2, :-2]) / 4
    # Frobenius norm of [[dIxx, dIxy],[dIxy, dIyy]] per interior pixel
    H_diff = np.sqrt(dIxx ** 2 + 2 * dIxy ** 2 + dIyy ** 2)
    return float(np.mean(H_diff))


def _extract_latents(model, loader, device: str, model_type: str, input_ps: int = 32,
                     log_map_fn=None, log_map_inv_fn=None) -> dict:
    """Run inference over *loader* and collect per-patch latents and reconstructions.

    Returns
    -------
    dict with keys:
        ``paths``             – list[str]
        ``conditions``        – list[int]
        ``annotation_labels`` – list[int]  (-1 = unlabelled)
        ``latents``           – np.ndarray (N, latent_dim)
        ``raws``              – list of np.ndarray (H, W) float32
        ``recons``            – list of np.ndarray (H, W) float32
    """
    model.eval()
    all_paths, all_conditions, all_ann_labels, all_ann_labels_2, all_latents = [], [], [], [], []
    all_raws, all_recons = [], []
    all_proj_latents = []   # z_proj from projection head (contrastive only)

    with torch.no_grad():
        for batch in loader:
            x_raw        = batch[0].to(device)  # (B, C, H, W) — original CIO-RB values
            conditions   = batch[1]             # int tensor
            ann_labels   = batch[2]             # int tensor (-1 = unlabelled)
            ann_labels_2 = batch[3]             # int tensor, second label or -1
            paths        = batch[4]             # list of str

            # Center-crop enlarged-context patches before model inference
            if x_raw.shape[-1] != input_ps:
                h, w = x_raw.shape[-2:]
                top  = (h - input_ps) // 2
                left = (w - input_ps) // 2
                x_raw = x_raw[:, :, top:top+input_ps, left:left+input_ps]

            # Apply log mapping (if enabled); model sees log-space values
            x = log_map_fn(x_raw) if log_map_fn is not None else x_raw

            if model_type == "ae":
                x_hat, z = model(x)
            elif model_type == "vae":
                x_hat, mu, _, _ = model(x)
                z = mu                          # use mean (deterministic)
            elif model_type == "semisup":
                x_hat, z, _ = model(x)
            else:  # contrastive
                x_hat, z = model(x)
                z_proj = model.project(z)
                all_proj_latents.append(z_proj.cpu().numpy())

            all_paths.extend(paths)
            all_conditions.extend(conditions.tolist())
            all_ann_labels.extend(ann_labels.tolist())
            all_ann_labels_2.extend(ann_labels_2.tolist())
            all_latents.append(z.cpu().numpy())

            # Inverse-map to CIO-RB space so metrics are in original intensity units
            raw_np   = x_raw.cpu().numpy()  # always original CIO-RB (B, C, H, W)
            if log_map_inv_fn is not None:
                recon_np = log_map_inv_fn(x_hat).cpu().numpy()
            else:
                recon_np = x_hat.cpu().numpy()
            for raw_patch, recon_patch in zip(raw_np, recon_np):
                if raw_patch.shape[0] == 1:   # single-channel → squeeze to (H, W)
                    raw_patch   = raw_patch[0]
                    recon_patch = recon_patch[0]
                all_raws.append(raw_patch.astype(np.float32))
                all_recons.append(recon_patch.astype(np.float32))

    # Per-patch reconstruction metrics
    all_mse, all_mean_intensity, all_norm_mse = [], [], []
    all_l1, all_hessian_l1 = [], []
    for raw_p, recon_p in zip(all_raws, all_recons):
        diff       = raw_p - recon_p
        mse        = float(np.mean(diff ** 2))
        l1         = float(np.mean(np.abs(diff)))
        mean_int   = float(np.mean(raw_p))
        norm_mse   = mse / mean_int if mean_int > 0 else float("nan")
        all_mse.append(mse)
        all_l1.append(l1)
        all_mean_intensity.append(mean_int)
        all_norm_mse.append(norm_mse)
        all_hessian_l1.append(_patch_hessian_l1(raw_p, recon_p))

    return {
        "paths":               all_paths,
        "conditions":          all_conditions,
        "annotation_labels":   all_ann_labels,
        "annotation_labels_2": all_ann_labels_2,
        "latents":             np.concatenate(all_latents, axis=0),
        "proj_latents":        np.concatenate(all_proj_latents, axis=0) if all_proj_latents else None,
        "raws":                all_raws,
        "recons":              all_recons,
        "recon_mse":           all_mse,
        "recon_l1":            all_l1,
        "mean_intensity":      all_mean_intensity,
        "norm_mse":            all_norm_mse,
        "recon_hessian_l1":    all_hessian_l1,
    }


def _save_latent_csv(
    train_result: dict,
    val_result: dict,
    datasets: list,
    label_order: list,
    result_dir: Path,
    label_order_2: list | None = None,
) -> Path:
    """Build and save the latent feature CSV.

    Columns
    -------
    filename, filepath, condition, condition_name, group,
    split, z_0 … z_{d-1},
    annotation_label, annotation_label_name,
    annotation_label_2, annotation_label_2_name  (when label_order_2 is given)
    """
    # condition → name mapping from loaded datasets
    cond_to_name = {}
    for ds in datasets:
        cond_to_name[ds.condition] = ds.condition_name

    int_to_label  = {i: lbl for i, lbl in enumerate(label_order)}  if label_order  else {}
    int_to_label2 = {i: lbl for i, lbl in enumerate(label_order_2)} if label_order_2 else {}
    has_label2    = bool(label_order_2)

    has_proj = train_result.get("proj_latents") is not None

    rows = []
    for split_name, result in [("train", train_result), ("val", val_result)]:
        latents      = result["latents"]
        proj_latents = result.get("proj_latents")
        ann2_list    = result.get("annotation_labels_2", [-1] * len(result["paths"]))
        for i, path in enumerate(result["paths"]):
            condition = result["conditions"][i]
            ann_int   = result["annotation_labels"][i]
            ann_int2  = ann2_list[i]
            row = {
                "filename":            Path(path).name,
                "filepath":            path,
                "condition":           condition,
                "condition_name":      cond_to_name.get(condition, str(condition)),
                "group":               _extract_group_key(path),
                "split":               split_name,
                "recon_mse":           result["recon_mse"][i],
                "recon_l1":            result["recon_l1"][i],
                "mean_intensity":      result["mean_intensity"][i],
                "norm_mse":            result["norm_mse"][i],
                "recon_hessian_l1":    result["recon_hessian_l1"][i],
                "annotation_label":    ann_int,
                "annotation_label_name": int_to_label.get(ann_int, ""),
            }
            if has_label2:
                row["annotation_label_2"]      = ann_int2
                row["annotation_label_2_name"] = int_to_label2.get(ann_int2, "")
            for d, val in enumerate(latents[i]):
                row[f"z_{d}"] = float(val)
            if has_proj and proj_latents is not None:
                for d, val in enumerate(proj_latents[i]):
                    row[f"p_{d}"] = float(val)
            rows.append(row)

    # column order: metadata, quality metrics, latent dims, proj dims, annotation
    latent_dim = result["latents"].shape[1]
    latent_cols = [f"z_{d}" for d in range(latent_dim)]
    meta_cols   = ["filename", "filepath", "condition", "condition_name",
                   "group", "split"]
    metric_cols = ["recon_mse", "recon_l1", "mean_intensity", "norm_mse", "recon_hessian_l1"]
    ann_cols    = ["annotation_label", "annotation_label_name"]
    if has_label2:
        ann_cols += ["annotation_label_2", "annotation_label_2_name"]
    proj_cols = []
    if has_proj and train_result["proj_latents"] is not None:
        proj_dim  = train_result["proj_latents"].shape[1]
        proj_cols = [f"p_{d}" for d in range(proj_dim)]
    df = pd.DataFrame(rows, columns=meta_cols + metric_cols + latent_cols + proj_cols + ann_cols)

    out_path = result_dir / "latents.csv"
    df.to_csv(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# Reconstruction output
# ---------------------------------------------------------------------------

def _save_reconstructions(
    train_result: dict,
    val_result: dict,
    cfg: "AEConfig",
    datasets: list | None = None,
) -> None:
    """Save per-patch and whole-image reconstruction outputs.

    Directory layout under ``cfg.result_dir / "recon"``::

        recon/
          patches/          # individual patch .tif files
            train_<name>.tif
            val_<name>.tif
          images/           # whole-image canvases
            raw_<group>.tif
            recon_<group>.tif
          visual/           # side-by-side PNG comparisons
            <group>_comparison.png

    Coordinates are parsed from the patch filename using
    :func:`_parse_patch_coords`.  ``cfg.recon_pad_size`` is subtracted from
    the padded coordinates to recover the original image positions.

    Parameters
    ----------
    train_result, val_result : dict
        Output of :func:`_extract_latents` (must include ``"raws"`` and
        ``"recons"`` keys).
    cfg : AEConfig
        Pipeline configuration; uses ``result_dir`` and ``recon_pad_size``.
    """
    recon_dir = cfg.result_dir / "recon"
    recon_dir.mkdir(parents=True, exist_ok=True)

    cond_to_name: dict = {}
    if datasets:
        for ds in datasets:
            cond_to_name[ds.condition] = ds.condition_name

    pad = cfg.recon_pad_size

    # Detect number of channels from the first available patch
    _all_patches = train_result["raws"] or val_result["raws"]
    n_ch = _all_patches[0].shape[0] if _all_patches and _all_patches[0].ndim == 3 else 1

    def _get_ch(arr, ch):
        """Return 2-D (H, W) slice for channel *ch* from a (C,H,W) or (H,W) array."""
        return arr[ch] if arr.ndim == 3 else arr

    # ---- 1. collect patches into lists, build canvas info ----
    # Patches are written as stacked TIFFs + a companion CSV:
    #   recon/patches_raw.tif   — (N, [C,] H, W) stack
    #   recon/patches_recon.tif — (N, [C,] H, W) stack
    #   recon/patches_index.csv — frame, split, condition, condition_name, group, name
    # canvas_data[group][split] = list of (y0, y1, x0, x1, raw, recon)
    canvas_data: dict = defaultdict(lambda: defaultdict(list))
    stack_raw:   list = []
    stack_recon: list = []
    stack_index: list = []   # rows for the CSV

    for split_name, result in [("train", train_result), ("val", val_result)]:
        for i, path in enumerate(result["paths"]):
            fname   = Path(path).name
            raw_p   = result["raws"][i]
            recon_p = result["recons"][i]

            condition      = result["conditions"][i]
            condition_name = cond_to_name.get(condition, str(condition))
            group          = _extract_group_key(path)
            stack_index.append({
                "frame":          len(stack_raw),
                "split":          split_name,
                "condition":      condition,
                "condition_name": condition_name,
                "group":          group,
                "name":           Path(fname).stem,
            })
            stack_raw.append(raw_p)
            stack_recon.append(recon_p)

            # parse coordinates for whole-image canvas
            coords = _parse_patch_coords(fname, path=path)
            if coords is None:
                continue  # skip if filename doesn't match pattern

            group, x_c, y_c, ps = coords
            half = ps // 2

            # original (unpadded) image coordinates
            r0 = y_c - half - pad
            r1 = y_c + half - pad
            c0 = x_c - half - pad
            c1 = x_c + half - pad

            if r0 < 0 or c0 < 0:
                log.debug("Skipping canvas placement for %s (negative coords)", fname)
                continue

            canvas_data[group][split_name].append((r0, r1, c0, c1, raw_p, recon_p))

    # ---- 1b. compact_recon: keep only 1 source image per (condition, split) ----
    if cfg.compact_recon:
        _seen_keys: set = set()
        _selected_groups: set = set()
        for row in stack_index:
            key = (row["condition_name"], row["split"])
            if key not in _seen_keys:
                _seen_keys.add(key)
                _selected_groups.add(row["group"])
        _kept_idx, _kept_raw, _kept_recon = [], [], []
        for row, raw_p, recon_p in zip(stack_index, stack_raw, stack_recon):
            if row["group"] in _selected_groups:
                _kept_idx.append({**row, "frame": len(_kept_idx)})
                _kept_raw.append(raw_p)
                _kept_recon.append(recon_p)
        stack_index, stack_raw, stack_recon = _kept_idx, _kept_raw, _kept_recon
        _compact_groups: set | None = _selected_groups & set(canvas_data.keys())
        log.info("compact_recon: keeping %d source images → %s",
                 len(_compact_groups), sorted(_compact_groups))
    else:
        _compact_groups = None

    # ---- 2. write stacked patch TIFFs + companion CSV ----
    if stack_raw:
        raw_stack   = np.stack(stack_raw,   axis=0)   # (N, [C,] H, W)
        recon_stack = np.stack(stack_recon, axis=0)
        if not cfg.compact_recon:
            tifffile.imwrite(str(recon_dir / "patches_raw.tif"), raw_stack, imagej=True)
        tifffile.imwrite(str(recon_dir / "patches_recon.tif"), recon_stack, imagej=True)
        idx_df = pd.DataFrame(stack_index)
        idx_df.to_csv(recon_dir / "patches_index.csv", index=False)
        log.info("Saved patch stacks (%d frames) → %s", len(stack_raw), recon_dir)

    # ---- 3. build whole-image canvases and visual comparisons per group ----
    # All outputs are stacked TIFFs + companion CSVs (no subdirectories):
    #   recon/images_raw.tif    — (N, H, W) stack, one frame per (group, channel)
    #   recon/images_recon.tif  — same
    #   recon/images_index.csv  — frame, group, channel
    #   recon/visual.tif        — (N, H, W, 3) uint8 stack, one frame per group
    #   recon/visual_index.csv  — frame, group
    img_size = cfg.recon_image_size
    all_groups = sorted(_compact_groups if _compact_groups is not None else set(canvas_data.keys()))

    img_stack_raw:   list = []
    img_stack_recon: list = []
    img_stack_index: list = []
    vis_stack:       list = []
    vis_index:       list = []

    for group in all_groups:
        split_entries = canvas_data[group]
        all_entries = [(r0, r1, c0, c1, raw_p, recon_p, sp)
                       for sp, entries in split_entries.items()
                       for (r0, r1, c0, c1, raw_p, recon_p) in entries]

        if not all_entries:
            continue

        # one canvas pair per channel
        raw_canvases   = [np.zeros((img_size, img_size), dtype=np.float32) for _ in range(n_ch)]
        recon_canvases = [np.zeros((img_size, img_size), dtype=np.float32) for _ in range(n_ch)]

        for r0, r1, c0, c1, raw_p, recon_p, _sp in all_entries:
            if r1 > img_size or c1 > img_size:
                log.warning(
                    "Patch [%d:%d, %d:%d] exceeds recon_image_size=%d for %s – clipping",
                    r0, r1, c0, c1, img_size, group,
                )
                r1 = min(r1, img_size)
                c1 = min(c1, img_size)
            for ch in range(n_ch):
                raw_canvases[ch][r0:r1, c0:c1]  = _get_ch(raw_p,   ch)[:r1-r0, :c1-c0]
                recon_canvases[ch][r0:r1, c0:c1] = _get_ch(recon_p, ch)[:r1-r0, :c1-c0]

        # collect canvas frames
        for ch in range(n_ch):
            img_stack_index.append({"frame": len(img_stack_raw), "group": group, "channel": ch})
            img_stack_raw.append(raw_canvases[ch])
            img_stack_recon.append(recon_canvases[ch])

        # render side-by-side comparison → numpy RGB array
        splits_present = sorted(split_entries.keys())
        suptitle = f"{group}  [{'+'.join(splits_present)}]"
        fig, axes = plt.subplots(n_ch, 2, figsize=(10, 5 * n_ch), squeeze=False)
        fig.suptitle(suptitle, fontsize=13, fontweight="bold")
        for ch in range(n_ch):
            ch_label = f" ch{ch}" if n_ch > 1 else ""
            axes[ch, 0].imshow(raw_canvases[ch],   cmap="gray", vmin=0, vmax=1)
            axes[ch, 0].set_title(f"Raw{ch_label}")
            axes[ch, 0].axis("off")
            axes[ch, 1].imshow(recon_canvases[ch], cmap="gray", vmin=0, vmax=1)
            axes[ch, 1].set_title(f"Reconstruction{ch_label}")
            axes[ch, 1].axis("off")
        fig.tight_layout()
        fig.canvas.draw()
        vis_w, vis_h = fig.canvas.get_width_height()
        vis_arr = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        vis_arr = vis_arr.reshape(vis_h, vis_w, 4)[:, :, :3]  # RGBA → RGB
        plt.close(fig)
        vis_index.append({"frame": len(vis_stack), "group": group})
        vis_stack.append(vis_arr)

    # ---- 4. write image and visual stacked TIFFs + CSVs ----
    if img_stack_raw:
        if not cfg.compact_recon:
            tifffile.imwrite(str(recon_dir / "images_raw.tif"),
                             np.stack(img_stack_raw, axis=0), imagej=True)
        tifffile.imwrite(str(recon_dir / "images_recon.tif"),
                         np.stack(img_stack_recon, axis=0), imagej=True)
        pd.DataFrame(img_stack_index).to_csv(recon_dir / "images_index.csv", index=False)
        log.info("Saved image canvas stacks (%d frames) → %s", len(img_stack_raw), recon_dir)

    if vis_stack:
        tifffile.imwrite(str(recon_dir / "visual.tif"),
                         np.stack(vis_stack, axis=0), imagej=True)
        pd.DataFrame(vis_index).to_csv(recon_dir / "visual_index.csv", index=False)
        log.info("Saved visual comparison stack (%d frames) → %s", len(vis_stack), recon_dir)

    log.info("Reconstruction output saved → %s  (%d source images)",
             recon_dir, len(all_groups))


# ---------------------------------------------------------------------------
# Public pipeline entry-point
# ---------------------------------------------------------------------------

def run_ae_pipeline(cfg: AEConfig):
    """Run the full autoencoder training pipeline.

    Parameters
    ----------
    cfg : AEConfig
        Fully-initialised configuration object.

    Returns
    -------
    torch.nn.Module
        The trained model (already moved back to CPU for safe serialisation).
    """
    # ------------------------------------------------------------------
    # 1. Log config summary
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("Autoencoder Training Pipeline")
    log.info("  result_dir    : %s", cfg.result_dir)
    log.info("  model_type    : %s", cfg.model_type)
    log.info("  patch_dirs    : %d director%s",
             len(cfg.patch_dirs), "y" if len(cfg.patch_dirs) == 1 else "ies")
    for entry in cfg.patch_dirs:
        if "channel_dirs" in entry:
            log.info("    channel_dirs=%s  condition=%s (%s)",
                     entry["channel_dirs"],
                     entry.get("condition", "?"),
                     entry.get("condition_name", ""))
        else:
            log.info("    path=%-50s  condition=%s (%s)",
                     entry.get("path", "?"),
                     entry.get("condition", entry.get("label", "?")),
                     entry.get("condition_name", ""))
    log.info("  latent_dim=%d  input_ps=%d  no_ch=%d  BN=%s  dropout=%s",
             cfg.latent_dim, cfg.input_ps, cfg.no_ch, cfg.BN_flag, cfg.dropout_flag)
    log.info("  epochs=%d  lr=%g  batch_size=%d  val_split=%.2f",
             cfg.epochs, cfg.lr, cfg.batch_size, cfg.val_split)
    log.info("  device        : %s", cfg.device)

    if cfg.model_type == "vae":
        log.info("  [VAE] out_activation=%s  beta=%.3f  beta_anneal=%s  recon_type=%s",
                 cfg.out_activation, cfg.beta, cfg.beta_anneal, cfg.recon_type)
    elif cfg.model_type == "semisup":
        if cfg.lambda_cls_2 > 0:
            log.info("  [SemiSup dual] num_classes=%d  num_classes_2=%d  "
                     "lambda_recon=%.3f  lambda_cls=%.3f  lambda_cls2=%.3f",
                     cfg.num_classes, cfg.num_classes_2,
                     cfg.lambda_recon, cfg.lambda_cls, cfg.lambda_cls_2)
        else:
            log.info("  [SemiSup] num_classes=%d  lambda_recon=%.3f  lambda_cls=%.3f",
                     cfg.num_classes, cfg.lambda_recon, cfg.lambda_cls)
        log.info("  [SemiSup reg] weight_decay=%g  early_stopping_patience=%d  min_epochs_for_best=%d",
                 cfg.weight_decay, cfg.early_stopping_patience, cfg.min_epochs_for_best)
    elif cfg.model_type in ("contrastive", "supcon"):
        log.info("  [%s] proj_dim=%d  noise_prob=%.3f  temperature=%.3f  "
                 "lambda_contrast=%.3f",
                 cfg.model_type.upper(),
                 cfg.proj_dim, cfg.noise_prob, cfg.temperature, cfg.lambda_contrast)

    log.info("=" * 60)

    # ------------------------------------------------------------------
    # 2. Build datasets
    # ------------------------------------------------------------------
    # Transform: expand (H, W) numpy array → (1, H, W) before the dataset
    # class wraps it in a torch tensor.  Patches are already float32 in [0, 1].
    def _channel_expand(x: np.ndarray) -> np.ndarray:
        out = np.expand_dims(x, 0)
        if cfg.patch_input_divisor != 1.0:
            out = out / cfg.patch_input_divisor
        return out

    def _load_hist_map(map_dir: str, patch_path: str):
        """Auto-detect dataset name from patch path and load its .npz map."""
        import re as _re
        m = _re.search(r"/cio_rb/(\w+)/", str(patch_path))
        if not m:
            return None
        ds_name = m.group(1)
        map_file = Path(map_dir) / f"{ds_name}_map.npz"
        if not map_file.exists():
            log.warning("hist_map_dir set but %s not found — skipping for %s",
                        map_file.name, ds_name)
            return None
        data = np.load(str(map_file))
        hmap = np.stack([data["src_q"], data["ref_q"]])  # (2, N)
        log.info("  Histogram map loaded: %s", map_file.name)
        return hmap

    datasets = []
    per_ds_val_splits_raw = []   # None or float per entry; None = use global cfg.val_split
    for entry in cfg.patch_dirs:
        condition      = int(entry.get("condition", entry.get("label", 0)))
        condition_name = str(entry.get("condition_name", str(condition)))

        shared_ann_kwargs = dict(
            annotation_file   = cfg.annotation_file or None,
            label_col         = cfg.label_col,
            filename_col      = cfg.filename_col,
            label_order       = cfg.label_order,
            annotation_file_2 = cfg.annotation_file_2 or None,
            label_col_2       = cfg.label_col_2,
            filename_col_2    = cfg.filename_col_2,
            label_order_2     = cfg.label_order_2,
        )

        # ── histogram map (optional) ──────────────────────────────────────
        hist_map = None
        if cfg.hist_map_dir:
            hist_map = _load_hist_map(
                cfg.hist_map_dir,
                entry.get("path") or (entry.get("channel_dirs") or [""])[0],
            )

        if "channel_dirs" in entry and cfg.enlarged_crop and "frame_dir" in entry:
            # Multi-channel enlarged crop: load context from per-channel source frames
            ch_list = cfg.enlarged_crop_channel  # must be a list, e.g. ["pax", "act"]
            if isinstance(ch_list, str):
                ch_list = [ch_list] * len(entry["channel_dirs"])
            ds = MultiChannelEnlargedCropDataset(
                patch_dirs=entry["channel_dirs"],
                frame_dir=entry["frame_dir"],
                channels=ch_list,
                condition=condition,
                condition_name=condition_name,
                context_size=cfg.enlarged_crop_context_size,
                patch_size=cfg.input_ps,
                pad_size=cfg.enlarged_crop_pad_size,
                input_divisor=cfg.enlarged_crop_input_divisor,
                input_clip_max=cfg.enlarged_crop_input_clip_max,
                **shared_ann_kwargs,
            )
        elif "channel_dirs" in entry:
            # Multi-channel: stack patches from each channel dir → (C, H, W)
            ds = MultiChannelPatchDataset(
                entry["channel_dirs"],
                condition=condition,
                condition_name=condition_name,
                **shared_ann_kwargs,
                # no transform: stacking already produces (C, H, W)
            )
        elif cfg.jitter_crop and "frame_dir" in entry:
            # On-the-fly jitter + rotation crop from source frame TIFFs
            ds = JitterCropDataset(
                patch_dir=entry["path"],
                frame_dir=entry["frame_dir"],
                channel=cfg.jitter_crop_channel,
                condition=condition,
                condition_name=condition_name,
                max_shift_px=cfg.jitter_crop_max_shift,
                max_angle_deg=cfg.jitter_crop_max_angle,
                patch_size=cfg.input_ps,
                pad_size=cfg.jitter_crop_pad_size,
                **shared_ann_kwargs,
            )
        elif cfg.enlarged_crop and "frame_dir" in entry:
            # Enlarged context crop — jitter+rotation applied per-view in training loop
            ds = EnlargedCropDataset(
                patch_dir=entry["path"],
                frame_dir=entry["frame_dir"],
                channel=cfg.enlarged_crop_channel,
                condition=condition,
                condition_name=condition_name,
                context_size=cfg.enlarged_crop_context_size,
                patch_size=cfg.input_ps,
                pad_size=cfg.enlarged_crop_pad_size,
                input_divisor=cfg.enlarged_crop_input_divisor,
                input_clip_max=cfg.enlarged_crop_input_clip_max,
                **shared_ann_kwargs,
            )
        else:
            ds = PatchDataset(
                entry["path"],
                condition=condition,
                condition_name=condition_name,
                **shared_ann_kwargs,
                transform=_channel_expand,
                hist_map=hist_map,
            )
        path_display = entry.get("path") or entry.get("channel_dirs", "?")
        if cfg.annotation_file and ds.num_classes > 0:
            cfg.num_classes = ds.num_classes
            log.info("  Loaded %d patches from %s  condition=%d (%s)  "
                     "annotation1: %d classes via %r",
                     len(ds), path_display, condition, condition_name,
                     ds.num_classes, cfg.label_col)
            log.info("  Label1 mapping: %s", ds.label_to_int)
        else:
            log.info("  Loaded %d patches from %s  condition=%d (%s)",
                     len(ds), path_display, condition, condition_name)
        if cfg.annotation_file_2 and ds.num_classes_2 > 0:
            cfg.num_classes_2 = ds.num_classes_2
            log.info("  annotation2: %d classes via %r  mapping: %s",
                     ds.num_classes_2, cfg.label_col_2, ds.label_to_int_2)
        per_ds_val_splits_raw.append(entry.get("val_split", None))
        datasets.append(ds)

    if not datasets:
        raise ValueError("patch_dirs is empty; nothing to train on.")

    full_dataset = ConcatDataset(datasets)
    total = len(full_dataset)
    log.info("  Total patches: %d", total)

    # ------------------------------------------------------------------
    # 3. Train / val split
    # ------------------------------------------------------------------
    if cfg.group_split:
        # Use per-dataset val_split if any entry specifies one; otherwise global.
        has_per_ds = any(v is not None for v in per_ds_val_splits_raw)
        per_ds_val_splits = (
            [v if v is not None else cfg.val_split for v in per_ds_val_splits_raw]
            if has_per_ds else None
        )
        train_indices, val_indices = _grouped_train_val_split(
            datasets, cfg.val_split, per_ds_val_splits=per_ds_val_splits
        )
        train_ds = Subset(full_dataset, train_indices)
        val_ds   = Subset(full_dataset, val_indices)
        all_paths = [p for ds in datasets for p in getattr(ds, "paths", [])]
        n_train_groups = len({_extract_group_key(all_paths[i]) for i in train_indices})
        n_val_groups   = len({_extract_group_key(all_paths[i]) for i in val_indices})
        log.info("  Group-aware split: %d train patches (%d images) / "
                 "%d val patches (%d images)",
                 len(train_indices), n_train_groups,
                 len(val_indices), n_val_groups)
    else:
        n_val   = max(1, int(total * cfg.val_split))
        n_train = total - n_val
        train_ds, val_ds = random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        log.info("  Random split: %d train / %d val", n_train, n_val)

    # ------------------------------------------------------------------
    # 4. DataLoaders
    # ------------------------------------------------------------------
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
    )

    # ------------------------------------------------------------------
    # 5. Instantiate model
    # ------------------------------------------------------------------
    if cfg.model_type == "ae":
        model = AE(
            latent_dim=cfg.latent_dim,
            input_ps=cfg.input_ps,
            no_ch=cfg.no_ch,
            BN_flag=cfg.BN_flag,
            dropout_flag=cfg.dropout_flag,
        )

    elif cfg.model_type == "vae":
        model = VAE32(
            in_channels=cfg.no_ch,
            latent_dim=cfg.latent_dim,
            out_activation=cfg.out_activation,
        )

    elif cfg.model_type == "semisup":
        model = SemiSupAE(
            num_classes=cfg.num_classes,
            latent_dim=cfg.latent_dim,
            input_ps=cfg.input_ps,
            no_ch=cfg.no_ch,
            BN_flag=cfg.BN_flag,
            dropout_flag=cfg.dropout_flag,
            num_classes_2=cfg.num_classes_2,
        )

    else:  # "contrastive" or "supcon"
        model = ContrastiveAE(
            latent_dim=cfg.latent_dim,
            proj_dim=cfg.proj_dim,
            input_ps=cfg.input_ps,
            no_ch=cfg.no_ch,
            noise_prob=cfg.noise_prob,
            BN_flag=cfg.BN_flag,
            output_sigmoid=cfg.output_sigmoid,
        )

    model = model.to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("  Model: %s  (%.2fM trainable parameters)",
             type(model).__name__, n_params / 1e6)

    # ------------------------------------------------------------------
    # 6. Train
    # ------------------------------------------------------------------

    # Build log-map callables (None when disabled)
    if cfg.log_map_enabled:
        _log_map_fn, _log_map_inv_fn = make_log_map(
            x_min=cfg.log_map_x_min,
            x_ref=cfg.log_map_x_ref,
            delta=cfg.log_map_delta,
        )
        log.info("  log_map enabled: x_min=%.3f  x_ref=%.1f  delta=%.2f",
                 cfg.log_map_x_min, cfg.log_map_x_ref, cfg.log_map_delta)
    else:
        _log_map_fn = _log_map_inv_fn = None

    result_dir_str = str(cfg.result_dir)

    _sched_kwargs = dict(
        lr_scheduler=cfg.lr_scheduler,
        lr_scheduler_patience=cfg.lr_scheduler_patience,
        lr_scheduler_factor=cfg.lr_scheduler_factor,
        lr_min=cfg.lr_min,
    )

    if cfg.model_type == "ae":
        model, _, _ = train_ae(
            model, train_loader, val_loader,
            device=cfg.device,
            epochs=cfg.epochs,
            lr=cfg.lr,
            loss_norm_flag=cfg.loss_norm_flag,
            result_dir=result_dir_str,
            weight_decay=cfg.weight_decay,
            enlarged_crop_max_shift=cfg.enlarged_crop_max_shift if cfg.enlarged_crop else 0,
            enlarged_crop_max_angle=cfg.enlarged_crop_max_angle if cfg.enlarged_crop else 0.0,
            patch_size=cfg.input_ps,
            **_sched_kwargs,
        )

    elif cfg.model_type == "vae":
        model, _, _ = train_vae(
            model, train_loader, val_loader,
            device=cfg.device,
            epochs=cfg.epochs,
            lr=cfg.lr,
            beta=cfg.beta,
            recon_type=cfg.recon_type,
            result_dir=result_dir_str,
            beta_anneal=cfg.beta_anneal,
            **_sched_kwargs,
        )

    elif cfg.model_type == "semisup":
        model, _, _ = train_semisup_ae(
            model, train_loader, val_loader,
            device=cfg.device,
            epochs=cfg.epochs,
            lr=cfg.lr,
            lambda_recon=cfg.lambda_recon,
            lambda_cls=cfg.lambda_cls,
            result_dir=result_dir_str,
            lambda_cls2=cfg.lambda_cls_2,
            weight_decay=cfg.weight_decay,
            early_stopping_patience=cfg.early_stopping_patience,
            min_epochs_for_best=cfg.min_epochs_for_best,
            warmup_epochs=cfg.warmup_epochs,
            lr_scheduler_patience=cfg.semisup_lr_scheduler_patience,
        )

    elif cfg.model_type == "contrastive":
        model, _, _ = train_contrastive_ae(
            model, train_loader, val_loader,
            device=cfg.device,
            epochs=cfg.epochs,
            lr=cfg.lr,
            lambda_recon=cfg.lambda_recon,
            lambda_contrast=cfg.lambda_contrast,
            result_dir=result_dir_str,
            noise_prob=cfg.noise_prob,
            temperature=cfg.temperature,
            use_flip=cfg.use_flip,
            weight_decay=cfg.weight_decay,
            warmup_epochs=cfg.warmup_epochs,
            min_epochs_for_best=cfg.min_epochs_for_best,
            lr_scheduler=cfg.lr_scheduler,
            patch_size=cfg.input_ps,
            enlarged_crop_max_shift=cfg.enlarged_crop_max_shift if cfg.enlarged_crop else 0,
            enlarged_crop_max_angle=cfg.enlarged_crop_max_angle if cfg.enlarged_crop else 0.0,
            recon_loss_type=cfg.recon_loss_type,
            lambda_hessian=cfg.lambda_hessian,
            log_map_fn=_log_map_fn,
            log_map_inv_fn=_log_map_inv_fn,
        )

    else:  # "supcon"
        model, _, _ = train_supervised_contrastive_ae(
            model, train_loader, val_loader,
            device=cfg.device,
            epochs=cfg.epochs,
            lr=cfg.lr,
            lambda_recon=cfg.lambda_recon,
            lambda_contrast=cfg.lambda_contrast,
            result_dir=result_dir_str,
            noise_prob=cfg.noise_prob,
            temperature=cfg.temperature,
            use_flip=cfg.use_flip,
            weight_decay=cfg.weight_decay,
            warmup_epochs=cfg.warmup_epochs,
            min_epochs_for_best=cfg.min_epochs_for_best,
            lr_scheduler=cfg.lr_scheduler,
            lr_min=cfg.lr_min,
            patch_size=cfg.input_ps,
            enlarged_crop_max_shift=cfg.enlarged_crop_max_shift if cfg.enlarged_crop else 0,
            enlarged_crop_max_angle=cfg.enlarged_crop_max_angle if cfg.enlarged_crop else 0.0,
            recon_loss_type=cfg.recon_loss_type,
            lambda_hessian=cfg.lambda_hessian,
            log_map_fn=_log_map_fn,
            log_map_inv_fn=_log_map_inv_fn,
        )

    # ------------------------------------------------------------------
    # 7. Save final model
    # ------------------------------------------------------------------
    final_path = cfg.result_dir / "model_final.pt"
    torch.save(model, str(final_path))
    log.info("Final model saved → %s", final_path)

    # ------------------------------------------------------------------
    # 8. Export latent CSV  (inference over full train + val sets)
    # ------------------------------------------------------------------
    log.info("Extracting latents for CSV export …")
    # Use shuffle=False loaders so row order is deterministic
    train_loader_ordered = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0
    )
    val_loader_ordered = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0
    )
    train_result = _extract_latents(model, train_loader_ordered, cfg.device, cfg.model_type,
                                    input_ps=cfg.input_ps,
                                    log_map_fn=_log_map_fn, log_map_inv_fn=_log_map_inv_fn)
    val_result   = _extract_latents(model, val_loader_ordered,   cfg.device, cfg.model_type,
                                    input_ps=cfg.input_ps,
                                    log_map_fn=_log_map_fn, log_map_inv_fn=_log_map_inv_fn)

    csv_path = _save_latent_csv(
        train_result, val_result,
        datasets=datasets,
        label_order=datasets[0].label_order if datasets else [],
        result_dir=cfg.result_dir,
        label_order_2=datasets[0].label_order_2 if (datasets and datasets[0].label_order_2) else None,
    )
    log.info("Latent CSV saved → %s  (%d rows)",
             csv_path, len(train_result["paths"]) + len(val_result["paths"]))

    # ------------------------------------------------------------------
    # 9. Save reconstruction images
    # ------------------------------------------------------------------
    if cfg.save_recon:
        log.info("Saving reconstruction images …")
        _save_reconstructions(train_result, val_result, cfg, datasets=datasets)

    log.info("Pipeline complete.")
    return model
