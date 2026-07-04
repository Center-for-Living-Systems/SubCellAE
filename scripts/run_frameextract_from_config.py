"""
run_frameextract_from_config.py
================================
Extract full-frame, per-channel CIO-RB normalized images from .czi files.

Usage
-----
    python scripts/run_frameextract_from_config.py config/frameextract_config/vinc_control_cio_rb.yaml
    python scripts/run_frameextract_from_config.py config/frameextract_config/vinc_control_cio_rb.yaml --debug
    python scripts/run_frameextract_from_config.py config/frameextract_config/vinc_control_cio_rb.yaml --root_folder /custom/data/root

The YAML file drives every setting; see FrameExtractConfig for field descriptions.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

try:
    import yaml
except ImportError:
    raise ImportError("PyYAML is required: pip install pyyaml")

from subcellae.pipeline.frameextract_pipeline import (
    ChannelExtractConfig,
    FrameExtractConfig,
    run_frameextract_pipeline,
)
from subcellae.utils.config_utils import resolve_root


def load_config(yaml_path: str | Path, root_folder: str | None = None) -> FrameExtractConfig:
    """Parse a YAML config file and return a :class:`FrameExtractConfig`."""
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    raw = resolve_root(raw, root_folder)

    paths = raw.get("paths", {})
    exp   = raw.get("experiment", {})
    inp   = raw.get("input", {})
    pre   = raw.get("preprocessing", {})
    seg   = raw.get("segmentation", {})
    misc  = raw.get("misc", {})

    channels = [
        ChannelExtractConfig(
            index=ch["index"],
            name=ch["name"],
            scale=float(ch.get("scale", 5.0)),
        )
        for ch in raw["channels"]
    ]

    return FrameExtractConfig(
        image_folder            = paths["image_folder"],
        output_dir              = paths["output_dir"],
        condition               = exp.get("condition", "unknown"),
        channels                = channels,
        cell_mask_folder        = paths.get("cell_mask_folder", None),
        rolling_ball_radius     = pre.get("rolling_ball_radius", None),
        seg_ch                  = seg.get("seg_ch", None),
        file_type               = inp.get("file_type", "czi"),
        start_ind               = inp.get("start_ind", 0),
        end_ind                 = inp.get("end_ind", 999),
        seg_threshold           = seg.get("seg_threshold", 0.1),
        seg_close_size          = seg.get("seg_close_size", 11),
        seg_min_size_initial    = seg.get("seg_min_size_initial", 3),
        seg_min_size_post_close = seg.get("seg_min_size_post_close", 10),
        seg_min_size_final      = seg.get("seg_min_size_final", 30000),
        debug                   = misc.get("debug", False),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract full-frame CIO-RB normalized channel images from CZI files."
    )
    parser.add_argument("config", help="Path to YAML config file.")
    parser.add_argument(
        "--root_folder",
        default=None,
        help="Override root_folder in the config (useful for cluster vs local paths).",
    )
    parser.add_argument("--debug", action="store_true", help="Stop after first file.")
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    cfg = load_config(args.config, root_folder=args.root_folder)
    if args.debug:
        cfg.debug = True

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    run_frameextract_pipeline(cfg)


if __name__ == "__main__":
    main()
