#!/usr/bin/env python3
"""
generate_conae_vinc_prt_configs.py

Convert existing CIO vinc contrastive/supcon configs to cio_mode_prt variants:
  - Replace source_frames/cio/ → source_frames/cio_mode_prt/
  - Set output_sigmoid: false
  - Update result_dir suffix cio_vinc → prt_vinc

Reads all ae_contrastive_cio_vinc_* and ae_supcon_cio_vinc_* from
config/contrastive_config/ and writes prt versions alongside.

Usage:
  python scripts/generate_conae_vinc_prt_configs.py
  python scripts/generate_conae_vinc_prt_configs.py --dry-run
"""
from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CFG_DIR   = REPO_ROOT / "config" / "contrastive_config"


def convert(text: str) -> str:
    text = text.replace(
        '"/ae_results/source_frames/cio/',
        '"/ae_results/source_frames/cio_mode_prt/',
    )
    text = text.replace(
        "output_sigmoid  : true",
        "output_sigmoid  : false",
    )
    text = text.replace(
        "_cio_vinc_",
        "_prt_vinc_",
    )
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    patterns = ["ae_contrastive_cio_vinc_*.yaml", "ae_supcon_cio_vinc_*.yaml"]
    src_files = []
    for pat in patterns:
        src_files.extend(sorted(CFG_DIR.glob(pat)))

    print(f"Converting {len(src_files)} configs → prt variants")

    for src in src_files:
        dst_name = src.name.replace("_cio_vinc_", "_prt_vinc_")
        dst      = src.parent / dst_name
        text     = convert(src.read_text())

        if args.dry_run:
            print(f"\n{'='*60}\n# {dst.name}")
            for line in text.splitlines()[:15]:
                print(line)
        else:
            dst.write_text(text)
            print(f"  wrote {dst.name}")


if __name__ == "__main__":
    main()
