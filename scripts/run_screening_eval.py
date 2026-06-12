"""
run_screening_eval.py
=====================
Apply a trained screening model to a new (unseen) dataset to test generalization.
No training — loads ``model_best.pt`` and runs inference on labelled patches.

Typical use: train on vinc, evaluate on ppax to check cross-dataset transfer.

Usage
-----
    python scripts/run_screening_eval.py \\
        config/screening_config/config_screening_ppax_eval.yaml

    # Override paths on the fly
    python scripts/run_screening_eval.py \\
        config/screening_config/config_screening_ppax_eval.yaml \\
        --model_pt /path/to/model_best.pt \\
        --out_dir  /path/to/output

Config keys
-----------
    model:
      model_pt   : "/path/to/model_best.pt"
      backbone   : "efficientnet_b0"   # must match the training config
      input_size : 224
      dropout    : 0.3

    data:  (same keys as config_screening_vinc.yaml)
      label_csv, patch_dirs, label_col, filename_col, condition_col,
      group_col, ad_labels, nonad_label, exclude_labels

    output:
      out_dir : "/path/to/eval_output"
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import yaml
except ImportError:
    raise ImportError("PyYAML is required: pip install pyyaml")

import numpy as np
import torch
from torch.utils.data import DataLoader

from subcellae.screening.dataset import (
    AD_LABELS, NONAD_LABEL,
    DatasetLinearCorrection, DatasetHistogramCorrection,
    PatchScreeningDataset, build_transforms,
    compute_dataset_stats, compute_histogram_correction, sample_dataset_pixels,
)
from subcellae.screening.evaluate import (
    compute_metrics,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_prob_histogram,
    plot_roc_curve,
    save_metrics_csv,
    save_metrics_txt,
    save_predictions_csv,
)
from subcellae.screening.model import ScreeningClassifier
from subcellae.screening.train import evaluate_epoch
from subcellae.utils.config_utils import resolve_root
import torch.nn as nn


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_eval_config(yaml_path: str | Path, root_folder: str | None = None) -> dict:
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config not found: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return resolve_root(raw, root_folder)


# ---------------------------------------------------------------------------
# Main eval function
# ---------------------------------------------------------------------------

def run_eval(
    *,
    model_pt:       str | Path,
    backbone:       str,
    input_size:     int,
    dropout:        float,
    label_csv:      str | Path,
    patch_dirs:     dict,
    out_dir:        str | Path,
    label_col:      str = "classification",
    filename_col:   str = "unique_ID",
    condition_col:  str = "condition",
    group_col:      str = "czi_filename",
    ad_labels:      list = None,
    nonad_label:    str = NONAD_LABEL,
    exclude_labels: list = None,
    batch_size:     int = 64,
    num_workers:    int = 4,
    device_str:     str = "auto",
    # dataset-level correction
    correction_type: str = "none",        # "none" | "linear" | "histogram"
    ref_patch_dirs: dict | None = None,   # training (reference) patch dirs
    max_stat_patches: int = 5000,
    ref_cdf_dir: str | None = None,       # dir with saved ref_cdf_*.npy (HM run)
) -> dict:
    """
    Parameters
    ----------
    correction_type : str
        ``"none"`` — no correction.
        ``"linear"`` — match mean and std only.
        ``"histogram"`` — full CDF-based histogram matching.
    ref_patch_dirs : dict, optional
        Patch directories of the *training* (reference) dataset (e.g. vinc).
        Required for ``"linear"`` and ``"histogram"`` unless ``ref_cdf_dir``
        is provided.
    ref_cdf_dir : str, optional
        Path to a training out_dir that contains ``ref_cdf_src.npy`` and
        ``ref_cdf_ref.npy`` (saved by the HM training pipeline).  When given,
        reuses those saved arrays instead of recomputing from patch files.
    """
    if ad_labels is None:
        ad_labels = list(AD_LABELS)
    if exclude_labels is None:
        exclude_labels = ["Uncertain"]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    log.info("Device: %s", device)

    # ------------------------------------------------------------------
    # 0. Build dataset-level correction
    # ------------------------------------------------------------------
    import json as _json, numpy as _np

    pixel_correction = None
    if correction_type != "none":
        log.info("Building '%s' correction …", correction_type)

        # ── reference pixels / CDF ───────────────────────────────────────
        if ref_cdf_dir is not None and correction_type == "histogram":
            # Reuse CDF saved by a previous HM training run
            cdf_dir = Path(ref_cdf_dir)
            ref_cdf_src = _np.load(str(cdf_dir / "ref_cdf_src.npy"))
            ref_cdf_ref = _np.load(str(cdf_dir / "ref_cdf_ref.npy"))
            log.info("  Loaded reference CDF from %s  (%d breakpoints)",
                     cdf_dir, len(ref_cdf_src))
            # Still need source pixels to build the src side of the mapping
            src_pix = sample_dataset_pixels(patch_dirs,
                                            max_patches=max_stat_patches)
            src_q   = _np.linspace(0.0, 1.0, len(ref_cdf_src))
            src_vals = _np.quantile(src_pix, src_q).astype(_np.float32)
            pixel_correction = DatasetHistogramCorrection(src_vals, ref_cdf_ref)

        elif correction_type == "histogram":
            if ref_patch_dirs is None:
                raise ValueError("ref_patch_dirs required for histogram correction")
            ref_pix = sample_dataset_pixels(ref_patch_dirs,
                                            max_patches=max_stat_patches)
            src_pix = sample_dataset_pixels(patch_dirs,
                                            max_patches=max_stat_patches)
            pixel_correction = compute_histogram_correction(
                ref_pix, src_pix, n_quantiles=2000
            )
            _np.save(str(out_dir / "ref_cdf_src.npy"), pixel_correction._src)
            _np.save(str(out_dir / "ref_cdf_ref.npy"), pixel_correction._ref)

        elif correction_type == "linear":
            if ref_patch_dirs is None:
                raise ValueError("ref_patch_dirs required for linear correction")
            ref_mean, ref_std = compute_dataset_stats(ref_patch_dirs,
                                                       max_patches=max_stat_patches)
            src_mean, src_std = compute_dataset_stats(patch_dirs,
                                                       max_patches=max_stat_patches)
            pixel_correction = DatasetLinearCorrection(ref_mean, ref_std,
                                                        src_mean, src_std)

        log.info("  correction: %s", pixel_correction)
        (out_dir / "dataset_correction_stats.json").write_text(
            _json.dumps({"correction_type": correction_type}, indent=2)
        )

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    log.info("Loading eval dataset …")
    transform = build_transforms(input_size, augment=False)
    ds = PatchScreeningDataset(
        label_csv=label_csv,
        patch_dirs=patch_dirs,
        label_col=label_col,
        filename_col=filename_col,
        condition_col=condition_col,
        group_col=group_col,
        ad_labels=ad_labels,
        nonad_label=nonad_label,
        exclude_labels=exclude_labels,
        pixel_correction=pixel_correction,
        transform=transform,
    )
    labels = ds.labels
    n_ad    = labels.sum()
    n_nonad = len(labels) - n_ad
    log.info("  Patches: %d  (adhesion=%d, no-adhesion=%d)", len(ds), n_ad, n_nonad)

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
    )

    # ------------------------------------------------------------------
    # 2. Load model
    # ------------------------------------------------------------------
    log.info("Loading model from %s …", model_pt)
    model = ScreeningClassifier(
        backbone=backbone,
        pretrained=False,    # weights come from checkpoint
        dropout=dropout,
        img_size=input_size,
    )
    state = torch.load(model_pt, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    log.info("  Model loaded (backbone=%s, input_size=%d)", backbone, input_size)

    # ------------------------------------------------------------------
    # 3. Inference
    # ------------------------------------------------------------------
    log.info("Running inference …")
    # Use a dummy pos_weight=1 for eval (no training, just evaluation)
    criterion = nn.BCEWithLogitsLoss()
    result = evaluate_epoch(model, loader, criterion, device)

    y_true  = result["all_labels"]
    y_proba = result["all_probs"]
    y_pred  = result["all_preds"]

    # ------------------------------------------------------------------
    # 4. Metrics and plots
    # ------------------------------------------------------------------
    log.info("Computing metrics …")
    metrics = compute_metrics(y_true, y_pred, y_proba)
    log.info("  accuracy          : %.4f", metrics["accuracy"])
    log.info("  balanced_accuracy : %.4f", metrics["balanced_accuracy"])
    log.info("  f1_macro          : %.4f", metrics["f1_macro"])
    log.info("  roc_auc           : %.4f", metrics["roc_auc"])
    log.info("\n%s", metrics["report"])

    save_metrics_txt(metrics, out_dir, split="eval")
    save_metrics_csv(metrics, out_dir)

    acc = metrics["accuracy"]
    plot_confusion_matrix(
        y_true, y_pred,
        normalize=False,
        title=f"Eval confusion matrix (counts)  acc={acc:.3f}",
        save_path=out_dir / "confusion_matrix_counts.png",
    )
    plot_confusion_matrix(
        y_true, y_pred,
        normalize=True,
        title=f"Eval confusion matrix (normalised)  acc={acc:.3f}",
        save_path=out_dir / "confusion_matrix_norm.png",
    )
    plot_roc_curve(
        y_true, y_proba,
        auc=metrics["roc_auc"],
        save_path=out_dir / "roc_curve.png",
    )
    plot_pr_curve(
        y_true, y_proba,
        ap=metrics["avg_precision"],
        save_path=out_dir / "pr_curve.png",
    )
    plot_prob_histogram(y_true, y_proba, save_path=out_dir / "prob_histogram.png")

    # ------------------------------------------------------------------
    # 5. Predictions CSV
    # ------------------------------------------------------------------
    save_predictions_csv(ds.df, y_pred, y_proba, out_dir)
    log.info("Eval complete.  Outputs → %s", out_dir)

    return {"metrics": metrics, "y_true": y_true, "y_pred": y_pred, "y_proba": y_proba}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained screening model on a new dataset (no training).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("config", help="Path to the eval YAML config.")
    p.add_argument("--model_pt",   default=None, help="Override model.model_pt.")
    p.add_argument("--out_dir",    default=None, help="Override output.out_dir.")
    p.add_argument("--backbone",   default=None,
                   help="Override model.backbone (must match training architecture).")
    p.add_argument("--input_size", type=int, default=None,
                   help="Override model.input_size (must match training resolution).")
    p.add_argument("--apply_correction", action="store_true", default=None,
                   help="Apply dataset-level linear correction. "
                        "Requires dataset_correction.ref_patch_dirs in config "
                        "or --ref_patch_dirs.")
    p.add_argument("--root_folder", default=None)
    p.add_argument("--log_level", default=None,
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, level.upper()),
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as fh:
        _raw_for_log = yaml.safe_load(fh)
    _setup_logging(args.log_level or _raw_for_log.get("misc", {}).get("log_level", "INFO"))

    log.info("Loading eval config: %s", args.config)
    raw = load_eval_config(args.config, root_folder=args.root_folder)

    def _get(section, key, default=None):
        return raw.get(section, {}).get(key, default)

    model_pt   = args.model_pt   or str(_get("model", "model_pt", ""))
    backbone   = args.backbone   or str(_get("model", "backbone",   "efficientnet_b0"))
    input_size = args.input_size or int(_get("model", "input_size", 224))
    dropout    = float(_get("model", "dropout",  0.3))

    label_csv     = str(_get("data", "label_csv",     ""))
    patch_dirs    = _get("data", "patch_dirs",    {}) or {}
    label_col     = str(_get("data", "label_col",     "classification"))
    filename_col  = str(_get("data", "filename_col",  "unique_ID"))
    condition_col = str(_get("data", "condition_col", "condition"))
    group_col     = str(_get("data", "group_col",     "czi_filename"))
    ad_labels     = list(_get("data", "ad_labels",    list(AD_LABELS)))
    nonad_label   = str(_get("data", "nonad_label",   NONAD_LABEL))
    exclude_labels = list(_get("data", "exclude_labels", ["Uncertain"]) or [])

    out_dir = args.out_dir or str(_get("output", "out_dir", "results/screening_eval"))
    batch_size  = int(_get("inference", "batch_size",  64))
    num_workers = int(_get("inference", "num_workers",  4))
    device_str  = str(_get("misc", "device", "auto"))

    # Dataset-level correction
    correction_type_cfg = str(_get("dataset_correction", "correction_type", "none"))
    ref_patch_dirs_cfg  = _get("dataset_correction", "ref_patch_dirs", None)
    max_stat_patches    = int(_get("dataset_correction", "max_stat_patches", 5000))
    ref_cdf_dir         = _get("dataset_correction", "ref_cdf_dir", None)

    # CLI flag --apply_correction forces histogram if not already set in config
    if args.apply_correction and correction_type_cfg == "none":
        correction_type_cfg = "histogram"
    correction_type = correction_type_cfg
    ref_patch_dirs  = ref_patch_dirs_cfg

    if not model_pt:
        raise ValueError("model.model_pt must be set in the config (or pass --model_pt)")
    if not label_csv:
        raise ValueError("data.label_csv must be set in the config")

    run_eval(
        model_pt=model_pt,
        backbone=backbone,
        input_size=input_size,
        dropout=dropout,
        label_csv=label_csv,
        patch_dirs=patch_dirs,
        out_dir=out_dir,
        label_col=label_col,
        filename_col=filename_col,
        condition_col=condition_col,
        group_col=group_col,
        ad_labels=ad_labels,
        nonad_label=nonad_label,
        exclude_labels=exclude_labels,
        batch_size=batch_size,
        num_workers=num_workers,
        device_str=device_str,
        correction_type=correction_type,
        ref_patch_dirs=ref_patch_dirs,
        max_stat_patches=max_stat_patches,
        ref_cdf_dir=ref_cdf_dir,
    )

    shutil.copy2(args.config, Path(out_dir) / Path(args.config).name)
    log.info("Done.")


if __name__ == "__main__":
    main()
