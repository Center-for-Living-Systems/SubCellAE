"""
run_screening_from_config.py
=============================
Load a YAML config and run the EfficientNet binary screening pipeline.

Usage
-----
    python scripts/run_screening_from_config.py config/screening_config/config_screening_vinc.yaml
    python scripts/run_screening_from_config.py config/screening_config/config_screening_vinc.yaml --dry_run
    python scripts/run_screening_from_config.py config/screening_config/config_screening_vinc.yaml --root_folder /my/data
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

# Ensure the repo root is on sys.path so `subcellae` resolves to this repo
# regardless of how (or from where) the script is invoked.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import yaml
except ImportError:
    raise ImportError("PyYAML is required: pip install pyyaml")

from subcellae.screening.pipeline import ScreeningConfig, run_screening_pipeline
from subcellae.utils.config_utils import resolve_root


# ---------------------------------------------------------------------------
# YAML → ScreeningConfig
# ---------------------------------------------------------------------------

def load_config(yaml_path: str | Path, root_folder: str | None = None) -> ScreeningConfig:
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    raw = resolve_root(raw, root_folder)

    def _get(section: str, key: str, default=None):
        return raw.get(section, {}).get(key, default)

    # ---- data ----
    label_csv     = Path(str(_get("data", "label_csv", "")))
    patch_dirs    = _get("data", "patch_dirs", {}) or {}
    label_col     = str(_get("data", "label_col",     "classification"))
    filename_col  = str(_get("data", "filename_col",  "unique_ID"))
    condition_col = str(_get("data", "condition_col", "condition"))
    group_col     = str(_get("data", "group_col",     "czi_filename"))
    ad_labels     = list(_get("data", "ad_labels",    [
        "Nascent Adhesion", "focal complex", "focal adhesion", "fibrillar adhesion"
    ]))
    nonad_label   = str(_get("data", "nonad_label",   "No adhesion"))
    exclude_labels = list(_get("data", "exclude_labels", ["Uncertain"]) or [])

    # ---- output ----
    out_dir = Path(str(_get("output", "out_dir", "results/screening")))

    # ---- model ----
    backbone   = str(_get("model", "backbone",   "efficientnet_b0"))
    pretrained = bool(_get("model", "pretrained", True))
    input_size = int(_get("model", "input_size",  224))
    dropout    = float(_get("model", "dropout",   0.3))

    # ---- training ----
    epochs           = int(_get("training",   "epochs",           50))
    batch_size       = int(_get("training",   "batch_size",       64))
    lr               = float(_get("training", "lr",               1e-3))
    weight_decay     = float(_get("training", "weight_decay",     0.01))
    lr_scheduler     = str(_get("training",   "lr_scheduler",     "cosine"))
    patience         = int(_get("training",   "patience",         15))
    use_augmentation = bool(_get("training",  "use_augmentation", True))
    num_workers      = int(_get("training",   "num_workers",      4))
    test_size        = float(_get("training", "test_size",        0.2))
    random_state     = int(_get("training",   "random_state",     42))

    # ---- misc ----
    device             = str(_get("misc",     "device",              "auto"))
    pixel_correction   = str(_get("training", "pixel_correction",    "none"))
    correction_max_patches = int(_get("training", "correction_max_patches", 5000))

    # ---- intensity jitter ----
    use_intensity_jitter = bool(_get("training", "use_intensity_jitter", False))
    jitter_scale_lo  = float(_get("training", "jitter_scale_lo", 0.5))
    jitter_scale_hi  = float(_get("training", "jitter_scale_hi", 2.0))
    jitter_shift_std = float(_get("training", "jitter_shift_std", 0.05))

    # ---- multi-channel ----
    # extra_patch_dirs: list of {condition: path} dicts; null/[] disables
    extra_patch_dirs_raw = _get("training", "extra_patch_dirs", None) or []
    extra_patch_dirs = [d for d in extra_patch_dirs_raw if d] if extra_patch_dirs_raw else []

    # ---- jitter crop ----
    jc = raw.get("jitter_crop", {}) or {}
    use_jitter_crop         = bool(jc.get("enabled",       False))
    jitter_crop_frame_dirs  = jc.get("frame_dirs",         None)  # {condition: path}
    jitter_crop_channel     = str(jc.get("channel",        "pax"))
    jitter_crop_max_shift   = int(jc.get("max_shift_px",   4))
    jitter_crop_max_angle   = float(jc.get("max_angle_deg", 15.0))
    jitter_crop_pad_size    = int(jc.get("pad_size",        64))

    # ---- gamma jitter ----
    gj = raw.get("gamma_jitter", {}) or {}
    use_gamma_jitter = bool(gj.get("enabled",  False))
    gamma_lo         = float(gj.get("gamma_lo", 0.4))
    gamma_hi         = float(gj.get("gamma_hi", 2.5))

    return ScreeningConfig(
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
        backbone=backbone,
        pretrained=pretrained,
        input_size=input_size,
        dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        lr_scheduler=lr_scheduler,
        patience=patience,
        use_augmentation=use_augmentation,
        num_workers=num_workers,
        test_size=test_size,
        random_state=random_state,
        device=device,
        pixel_correction=pixel_correction,
        correction_max_patches=correction_max_patches,
        use_intensity_jitter=use_intensity_jitter,
        jitter_scale_range=(jitter_scale_lo, jitter_scale_hi),
        jitter_shift_std=jitter_shift_std,
        extra_patch_dirs=extra_patch_dirs if extra_patch_dirs else None,
        use_jitter_crop=use_jitter_crop,
        jitter_crop_frame_dirs=jitter_crop_frame_dirs,
        jitter_crop_channel=jitter_crop_channel,
        jitter_crop_max_shift=jitter_crop_max_shift,
        jitter_crop_max_angle=jitter_crop_max_angle,
        jitter_crop_pad_size=jitter_crop_pad_size,
        use_gamma_jitter=use_gamma_jitter,
        gamma_range=(gamma_lo, gamma_hi),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the binary screening pipeline from a YAML config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("config", help="Path to the YAML configuration file.")
    p.add_argument(
        "--dry_run", action="store_true",
        help="Print the resolved ScreeningConfig and exit without training.",
    )
    p.add_argument(
        "--log_level", default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    p.add_argument(
        "--root_folder", default=None,
        help="Override root_folder for all paths.",
    )
    # ---- sweep overrides (override corresponding YAML values) ----
    p.add_argument(
        "--backbone", default=None,
        help="timm backbone name (e.g. resnet18, vit_tiny_patch16_224). "
             "Overrides model.backbone in the config.",
    )
    p.add_argument(
        "--input_size", type=int, default=None,
        help="Input spatial resolution in pixels (e.g. 64, 128, 224). "
             "Overrides model.input_size in the config.",
    )
    p.add_argument(
        "--out_dir", default=None,
        help="Output directory. Overrides output.out_dir in the config. "
             "Useful for sweeps so each run lands in its own folder.",
    )
    p.add_argument(
        "--pixel_correction", default=None,
        choices=["none", "histogram", "linear"],
        help="Override training.pixel_correction in the config.",
    )
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
        _raw = yaml.safe_load(fh)
    yaml_log_level      = _raw.get("misc", {}).get("log_level", "INFO")
    effective_log_level = args.log_level or yaml_log_level
    _setup_logging(effective_log_level)

    log = logging.getLogger(__name__)
    log.info("Loading config from: %s", args.config)

    cfg = load_config(args.config, root_folder=args.root_folder)

    # Apply CLI overrides
    if args.backbone is not None:
        log.info("CLI override: backbone = %s", args.backbone)
        cfg.backbone = args.backbone
    if args.input_size is not None:
        log.info("CLI override: input_size = %d", args.input_size)
        cfg.input_size = args.input_size
    if args.out_dir is not None:
        log.info("CLI override: out_dir = %s", args.out_dir)
        cfg.out_dir = Path(args.out_dir)
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
    if args.pixel_correction is not None:
        log.info("CLI override: pixel_correction = %s", args.pixel_correction)
        cfg.pixel_correction = args.pixel_correction

    if args.dry_run:
        print("\n=== DRY RUN – resolved ScreeningConfig ===")
        for k, v in vars(cfg).items():
            print(f"  {k:<25} {v}")
        print("\nNo training performed.  Remove --dry_run to run for real.")
        return

    run_screening_pipeline(cfg)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, cfg.out_dir / Path(args.config).name)
    log.info("Config copied to: %s", cfg.out_dir)
    log.info("Done.")


if __name__ == "__main__":
    main()
