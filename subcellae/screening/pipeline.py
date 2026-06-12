"""
pipeline.py
===========
End-to-end pipeline: load data → split → train EfficientNet → evaluate → save outputs.

Outputs written to ``out_dir``:
  model_best.pt                – best checkpoint (by val loss)
  metrics.txt                  – text summary of val metrics
  metrics.csv                  – per-class precision/recall/F1
  loss_curves.png              – train/val loss and accuracy curves
  confusion_matrix_counts.png  – raw-count heatmap (val)
  confusion_matrix_norm.png    – normalised heatmap (val)
  roc_curve.png                – ROC curve with AUC
  pr_curve.png                 – precision-recall curve with AP
  prob_histogram.png           – P(adhesion) distribution by true class
  predictions_all.csv          – per-patch predictions for all labelled patches
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Subset

from .dataset import (
    AD_LABELS,
    NONAD_LABEL,
    DatasetHistogramCorrection,
    DatasetLinearCorrection,
    GammaJitter,
    IntensityJitter,
    JitterCropAugmentation,
    PatchScreeningDataset,
    build_transforms,
    compute_histogram_correction,
    compute_dataset_stats,
    sample_dataset_pixels,
)
from .evaluate import (
    compute_metrics,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_prob_histogram,
    plot_roc_curve,
    plot_training_curves,
    save_metrics_csv,
    save_metrics_txt,
    save_predictions_csv,
)
from .model import ScreeningClassifier
from .train import compute_pos_weight, train

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScreeningConfig:
    """All parameters for one screening run.

    Parameters
    ----------
    label_csv : Path
        Label CSV with per-patch classifications.
    patch_dirs : dict[str, str]
        Mapping condition → directory of patch TIFFs.
    out_dir : Path
        Where all outputs are written.

    Label columns
    -------------
    label_col : str
    filename_col : str
    condition_col : str
    group_col : str
        Column for group-aware train/val split (e.g. ``"czi_filename"``).
    ad_labels : list[str]
        Labels to group as class 1 (adhesion).
    nonad_label : str
        Label for class 0 (no adhesion).
    exclude_labels : list[str]
        Labels to drop entirely (e.g. ``["Uncertain"]``).

    Model
    -----
    backbone : str
        timm model name.
    pretrained : bool
    input_size : int
        Spatial resolution fed to the network.
    dropout : float

    Training
    --------
    epochs, batch_size, lr, weight_decay, lr_scheduler,
    patience, use_augmentation, num_workers, test_size, random_state
    """

    # --- required ---
    label_csv:  Path
    patch_dirs: dict
    out_dir:    Path

    # --- label columns ---
    label_col:     str           = "classification"
    filename_col:  str           = "unique_ID"
    condition_col: str           = "condition"
    group_col:     str           = "czi_filename"
    ad_labels:     list          = field(default_factory=lambda: list(AD_LABELS))
    nonad_label:   str           = NONAD_LABEL
    exclude_labels: list         = field(default_factory=lambda: ["Uncertain"])

    # --- model ---
    backbone:   str   = "efficientnet_b0"
    pretrained: bool  = True
    input_size: int   = 224
    dropout:    float = 0.3

    # --- training ---
    epochs:          int   = 50
    batch_size:      int   = 64
    lr:              float = 1e-3
    weight_decay:    float = 0.01
    lr_scheduler:    str   = "cosine"   # "cosine" | "plateau" | "none"
    patience:        int   = 15
    use_augmentation: bool = True
    num_workers:     int   = 4
    test_size:       float = 0.2
    random_state:    int   = 42

    # --- device ---
    device: str = "auto"   # "auto" | "cpu" | "cuda"

    # --- pixel correction applied at load time ---
    pixel_correction: str = "none"       # "none" | "histogram" | "linear"
    correction_max_patches: int = 5000

    # --- intensity jitter (training augmentation) ---
    # Applied as a pixel-level multiplicative scale + additive shift BEFORE
    # clipping to [0,1].  Simulates inter-channel and inter-dataset intensity
    # variation so the model becomes robust to distribution shift.
    use_intensity_jitter: bool = False
    jitter_scale_range: tuple  = (0.5, 2.0)   # multiplicative U(lo, hi)
    jitter_shift_std: float    = 0.05          # additive N(0, std)

    # --- multi-channel training ---
    # Additional patch directories (list of {condition: dir} dicts) to include
    # as separate training samples.  Labels from the primary label_csv are
    # re-used for each extra channel (same binary label, different patch image).
    # Leave as None/[] to train on a single channel.
    extra_patch_dirs: list = None

    # --- jitter-crop augmentation (on-the-fly from source frames) ---
    # When enabled, patches are re-cropped from full-frame source TIFFs each
    # epoch with a small random translation (±max_shift_px) and rotation
    # (±max_angle_deg).  Requires pre-extracted source frames at frame_dirs.
    use_jitter_crop: bool = False
    jitter_crop_frame_dirs: dict = None   # {condition: frame_dir_path}
    jitter_crop_channel: str = "pax"
    jitter_crop_max_shift: int = 4
    jitter_crop_max_angle: float = 15.0
    jitter_crop_pad_size: int = 64

    # --- gamma jitter (non-linear histogram augmentation) ---
    use_gamma_jitter: bool = False
    gamma_range: tuple = (0.4, 2.5)

    def __post_init__(self):
        self.label_csv = Path(self.label_csv)
        self.out_dir   = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.patch_dirs = {k: str(v) for k, v in self.patch_dirs.items()}
        if self.pixel_correction not in {"none", "histogram", "linear"}:
            raise ValueError(
                f"pixel_correction must be 'none', 'histogram', or 'linear', "
                f"got {self.pixel_correction!r}"
            )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_screening_pipeline(cfg: ScreeningConfig) -> dict:
    """Run the full EfficientNet binary screening pipeline."""

    log.info("=" * 60)
    log.info("Screening Pipeline  (EfficientNet binary classifier)")
    log.info("  label_csv   : %s", cfg.label_csv)
    log.info("  out_dir     : %s", cfg.out_dir)
    log.info("  backbone    : %s  (pretrained=%s)", cfg.backbone, cfg.pretrained)
    log.info("  input_size  : %d", cfg.input_size)
    log.info("  epochs      : %d  lr=%.4f", cfg.epochs, cfg.lr)
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)
    log.info("Device: %s", device)

    # ------------------------------------------------------------------
    # 0. Build pixel correction (if requested)
    # ------------------------------------------------------------------
    pixel_correction = None
    if cfg.pixel_correction != "none":
        log.info("Step 0: Building pixel correction ('%s') …", cfg.pixel_correction)
        # The training dataset is its own reference — we normalise it to its
        # own distribution so the pipeline is consistent when new datasets
        # (e.g. ppax) are corrected at inference to the same reference.
        ref_pix = sample_dataset_pixels(cfg.patch_dirs,
                                        max_patches=cfg.correction_max_patches)
        if cfg.pixel_correction == "histogram":
            pixel_correction = compute_histogram_correction(
                ref_pix, ref_pix, n_quantiles=2000
            )
            # self-correction is identity — but establishes the reference CDF
            # that inference scripts will use (saved to out_dir).
        elif cfg.pixel_correction == "linear":
            m, s = float(ref_pix.mean()), float(ref_pix.std())
            pixel_correction = DatasetLinearCorrection(m, s, m, s)
            # identity for training data; reference stats are saved below
        log.info("  correction: %s", pixel_correction)

        import json, numpy as _np
        _stats = {
            "pixel_correction": cfg.pixel_correction,
            "ref_mean": float(ref_pix.mean()),
            "ref_std":  float(ref_pix.std()),
        }
        if cfg.pixel_correction == "histogram":
            _np.save(str(cfg.out_dir / "ref_cdf_src.npy"),
                     pixel_correction._src)
            _np.save(str(cfg.out_dir / "ref_cdf_ref.npy"),
                     pixel_correction._ref)
            _stats["ref_cdf_src"] = "ref_cdf_src.npy"
            _stats["ref_cdf_ref"] = "ref_cdf_ref.npy"
        (cfg.out_dir / "correction_stats.json").write_text(
            json.dumps(_stats, indent=2)
        )
        log.info("  Correction stats saved → %s", cfg.out_dir / "correction_stats.json")

    # ------------------------------------------------------------------
    # 0b. Build intensity jitter (training only)
    # ------------------------------------------------------------------
    intensity_jitter_obj = None
    if cfg.use_intensity_jitter:
        intensity_jitter_obj = IntensityJitter(
            scale_range=tuple(cfg.jitter_scale_range),
            shift_std=cfg.jitter_shift_std,
        )
        log.info("Intensity jitter enabled: %s", intensity_jitter_obj)

    # ------------------------------------------------------------------
    # 0b2. Build gamma jitter (training only)
    # ------------------------------------------------------------------
    gamma_jitter_obj = None
    if cfg.use_gamma_jitter:
        gamma_jitter_obj = GammaJitter(gamma_range=cfg.gamma_range)
        log.info("Gamma jitter enabled: %s", gamma_jitter_obj)

    # ------------------------------------------------------------------
    # 0c. Build jitter-crop augmentation (training only)
    # ------------------------------------------------------------------
    jitter_crop_obj = None
    if cfg.use_jitter_crop:
        if not cfg.jitter_crop_frame_dirs:
            raise ValueError("use_jitter_crop=True but jitter_crop_frame_dirs is empty")
        jitter_crop_obj = JitterCropAugmentation(
            frame_dirs=cfg.jitter_crop_frame_dirs,
            channel=cfg.jitter_crop_channel,
            max_shift_px=cfg.jitter_crop_max_shift,
            max_angle_deg=cfg.jitter_crop_max_angle,
            pad_size=cfg.jitter_crop_pad_size,
        )
        log.info("Jitter crop enabled: %s", jitter_crop_obj)

    extra_dirs = cfg.extra_patch_dirs or []
    if extra_dirs:
        log.info("Multi-channel training: %d extra channel dir(s)", len(extra_dirs))

    # ------------------------------------------------------------------
    # 1. Build full dataset (no transform yet — we'll add after splitting)
    # ------------------------------------------------------------------
    log.info("Step 1: Loading primary-channel dataset for splitting …")
    # full_ds is built from the PRIMARY channel only so that train/val split
    # indices consistently refer to patch locations, not channel copies.
    # Extra channels are added only inside train_ds (after the split).
    full_ds = PatchScreeningDataset(
        label_csv=cfg.label_csv,
        patch_dirs=cfg.patch_dirs,
        label_col=cfg.label_col,
        filename_col=cfg.filename_col,
        condition_col=cfg.condition_col,
        group_col=cfg.group_col,
        ad_labels=cfg.ad_labels,
        nonad_label=cfg.nonad_label,
        exclude_labels=cfg.exclude_labels,
        # NO extra_patch_dirs here — split must be on primary locations only
    )

    labels = full_ds.labels
    groups = full_ds.groups
    n_ad    = labels.sum()
    n_nonad = len(labels) - n_ad
    log.info("  Total labelled patches: %d  (adhesion=%d, no-adhesion=%d)", len(labels), n_ad, n_nonad)
    log.info("  Unique groups (%s): %d", cfg.group_col, len(np.unique(groups)))

    # ------------------------------------------------------------------
    # 2. Group-aware train/val split
    # ------------------------------------------------------------------
    log.info("Step 2: Group-aware train/val split (test_size=%.2f) …", cfg.test_size)
    gss = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.random_state)
    train_idx, val_idx = next(gss.split(np.arange(len(full_ds)), labels, groups))
    log.info("  Train: %d  |  Val: %d", len(train_idx), len(val_idx))
    log.info("  Train label dist — adhesion: %d  no-adhesion: %d",
             labels[train_idx].sum(), (1 - labels[train_idx]).sum())
    log.info("  Val   label dist — adhesion: %d  no-adhesion: %d",
             labels[val_idx].sum(), (1 - labels[val_idx]).sum())

    # ------------------------------------------------------------------
    # 3. Build train/val datasets with their respective transforms
    # ------------------------------------------------------------------
    train_transform = build_transforms(cfg.input_size, augment=cfg.use_augmentation)
    val_transform   = build_transforms(cfg.input_size, augment=False)

    # Re-create datasets with indices so transforms apply correctly
    train_ds = PatchScreeningDataset(
        label_csv=cfg.label_csv,
        patch_dirs=cfg.patch_dirs,
        label_col=cfg.label_col,
        filename_col=cfg.filename_col,
        condition_col=cfg.condition_col,
        group_col=cfg.group_col,
        ad_labels=cfg.ad_labels,
        nonad_label=cfg.nonad_label,
        exclude_labels=cfg.exclude_labels,
        extra_patch_dirs=extra_dirs if extra_dirs else None,
        pixel_correction=pixel_correction,
        intensity_jitter=intensity_jitter_obj,   # training only
        gamma_jitter=gamma_jitter_obj,           # training only
        jitter_crop=jitter_crop_obj,             # training only
        transform=train_transform,
        indices=train_idx,
    )
    val_ds = PatchScreeningDataset(
        label_csv=cfg.label_csv,
        patch_dirs=cfg.patch_dirs,
        label_col=cfg.label_col,
        filename_col=cfg.filename_col,
        condition_col=cfg.condition_col,
        group_col=cfg.group_col,
        ad_labels=cfg.ad_labels,
        nonad_label=cfg.nonad_label,
        exclude_labels=cfg.exclude_labels,
        # val: primary channel only, no jitter → clean evaluation
        pixel_correction=pixel_correction,
        transform=val_transform,
        indices=val_idx,
    )
    n_primary_train = labels[train_idx].shape[0]
    log.info("  Primary locations — train: %d  val: %d",
             n_primary_train, labels[val_idx].shape[0])
    log.info("  Train dataset after channel expansion: %d samples  "
             "(%d locations × %d channel(s))",
             len(train_ds), n_primary_train,
             len(train_ds) // max(n_primary_train, 1))
    log.info("  Val dataset (primary channel only): %d samples", len(val_ds))

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"),
    )

    # ------------------------------------------------------------------
    # 4. Build model
    # ------------------------------------------------------------------
    log.info("Step 4: Building model (%s, pretrained=%s, img_size=%d) …",
             cfg.backbone, cfg.pretrained, cfg.input_size)
    model = ScreeningClassifier(
        backbone=cfg.backbone,
        pretrained=cfg.pretrained,
        dropout=cfg.dropout,
        img_size=cfg.input_size,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("  Trainable parameters: %s", f"{n_params:,}")

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    log.info("Step 5: Training …")
    pos_weight = compute_pos_weight(train_ds.labels, device)
    log.info("  pos_weight (no-ad/ad): %.3f", pos_weight.item())

    best_model_path = str(cfg.out_dir / "model_best.pt")
    result = train(
        model, train_loader, val_loader,
        epochs=cfg.epochs,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        lr_scheduler=cfg.lr_scheduler,
        pos_weight=pos_weight,
        device=device,
        save_best_path=best_model_path,
        patience=cfg.patience,
    )
    history    = result["history"]
    best_epoch = result["best_epoch"]
    log.info("  Training complete. Best epoch: %d", best_epoch)

    # Reload best weights for final evaluation
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    log.info("  Best model loaded from %s", best_model_path)

    # ------------------------------------------------------------------
    # 6. Evaluate on validation set
    # ------------------------------------------------------------------
    log.info("Step 6: Evaluating on validation set …")
    from .train import evaluate_epoch
    import torch.nn as nn
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    val_result = evaluate_epoch(model, val_loader, criterion, device)

    y_true  = val_result["all_labels"]
    y_proba = val_result["all_probs"]
    y_pred  = val_result["all_preds"]

    val_metrics = compute_metrics(y_true, y_pred, y_proba)
    log.info("  accuracy          : %.4f", val_metrics["accuracy"])
    log.info("  balanced_accuracy : %.4f", val_metrics["balanced_accuracy"])
    log.info("  f1_macro          : %.4f", val_metrics["f1_macro"])
    log.info("  roc_auc           : %.4f", val_metrics["roc_auc"])
    log.info("\n%s", val_metrics["report"])

    # ------------------------------------------------------------------
    # 7. Evaluate on training set (to check overfitting)
    # ------------------------------------------------------------------
    train_loader_noaug = DataLoader(
        PatchScreeningDataset(
            label_csv=cfg.label_csv,
            patch_dirs=cfg.patch_dirs,
            label_col=cfg.label_col,
            filename_col=cfg.filename_col,
            condition_col=cfg.condition_col,
            group_col=cfg.group_col,
            ad_labels=cfg.ad_labels,
            nonad_label=cfg.nonad_label,
            exclude_labels=cfg.exclude_labels,
            pixel_correction=pixel_correction,
            transform=val_transform,
            indices=train_idx,
        ),
        batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers,
    )
    train_result = evaluate_epoch(model, train_loader_noaug, criterion, device)
    train_metrics = compute_metrics(
        train_result["all_labels"], train_result["all_preds"], train_result["all_probs"]
    )
    log.info("  [Train] accuracy=%.4f  balanced_acc=%.4f  roc_auc=%.4f",
             train_metrics["accuracy"], train_metrics["balanced_accuracy"], train_metrics["roc_auc"])

    # ------------------------------------------------------------------
    # 8. Save metrics and plots
    # ------------------------------------------------------------------
    log.info("Step 8: Saving metrics and plots …")

    save_metrics_txt(val_metrics, cfg.out_dir)
    save_metrics_csv(val_metrics, cfg.out_dir)

    plot_training_curves(history, save_path=cfg.out_dir / "loss_curves.png")

    acc = val_metrics["accuracy"]
    plot_confusion_matrix(
        y_true, y_pred,
        normalize=False,
        title=f"Val confusion matrix (counts)  acc={acc:.3f}",
        save_path=cfg.out_dir / "confusion_matrix_counts.png",
    )
    plot_confusion_matrix(
        y_true, y_pred,
        normalize=True,
        title=f"Val confusion matrix (normalised)  acc={acc:.3f}",
        save_path=cfg.out_dir / "confusion_matrix_norm.png",
    )
    plot_roc_curve(
        y_true, y_proba,
        auc=val_metrics["roc_auc"],
        save_path=cfg.out_dir / "roc_curve.png",
    )
    plot_pr_curve(
        y_true, y_proba,
        ap=val_metrics["avg_precision"],
        save_path=cfg.out_dir / "pr_curve.png",
    )
    plot_prob_histogram(y_true, y_proba, save_path=cfg.out_dir / "prob_histogram.png")

    # ------------------------------------------------------------------
    # 9. Predictions for all labelled patches
    # ------------------------------------------------------------------
    log.info("Step 9: Predictions for all labelled patches …")
    all_ds = PatchScreeningDataset(
        label_csv=cfg.label_csv,
        patch_dirs=cfg.patch_dirs,
        label_col=cfg.label_col,
        filename_col=cfg.filename_col,
        condition_col=cfg.condition_col,
        group_col=cfg.group_col,
        ad_labels=cfg.ad_labels,
        nonad_label=cfg.nonad_label,
        exclude_labels=cfg.exclude_labels,
        pixel_correction=pixel_correction,
        transform=val_transform,
    )
    all_loader = DataLoader(
        all_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers,
    )
    all_result = evaluate_epoch(model, all_loader, criterion, device)

    # Tag each row with its split (train / val)
    # Use dtype=object so "train" is not truncated to "tra"
    split_tags = np.full(len(full_ds), "val", dtype=object)
    split_tags[train_idx] = "train"
    # Align: full_ds and all_ds share the same underlying df (same filtering)
    all_ds._df["split"] = split_tags

    save_predictions_csv(
        all_ds.df,
        all_result["all_preds"],
        all_result["all_probs"],
        cfg.out_dir,
        split_col="split",
    )
    log.info("  predictions_all.csv saved (%d rows)", len(all_ds))

    log.info("Screening pipeline complete.  Outputs → %s", cfg.out_dir)
    return {
        "model":       model,
        "val_metrics": val_metrics,
        "history":     history,
        "best_epoch":  best_epoch,
    }
