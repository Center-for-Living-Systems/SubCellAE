#!/usr/bin/env python3
"""
Merge FA-type and position annotation labels into existing latents.csv files,
then regenerate recon metric plots (MSE / L1 / Hessian L1) broken down by
FA type × train/val split.

Reads:
  <label_csv>        — labels_vinc_20260521.csv (unique_ID, classification, Position)
  <run_dir>/*/latents.csv

Writes (in-place):
  <run_dir>/*/latents.csv  — adds annotation_label, annotation_label_name,
                              annotation_label_2, annotation_label_2_name columns

Then calls add_recon_metrics.process_variant() for each variant.

Usage:
  python scripts/add_fa_labels_to_latents.py <run_dir>
  python scripts/add_fa_labels_to_latents.py <run_dir> --label-csv /path/to/labels.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from add_recon_metrics import process_variant   # reuse existing plot logic


LABEL_CSV_DEFAULT = (
    "/net/projects/CLS/lding/data/fa_data_analysis/labelling/labels_vinc_20260521.csv"
)

FA_ORDER  = ["Nascent Adhesion", "focal complex", "focal adhesion",
             "fibrillar adhesion", "No adhesion"]
POS_ORDER = ["Cell Protruding Edge", "Cell Periphery/other", "Lamella", "Cell Body"]


def _build_label_maps(label_csv: Path):
    lab = pd.read_csv(label_csv)
    lab["_key"] = lab["unique_ID"].apply(lambda p: Path(p).stem)

    fa_map  = dict(zip(lab["_key"], lab["classification"]))
    pos_map = dict(zip(lab["_key"], lab["Position"]))

    fa_idx  = {v: i for i, v in enumerate(FA_ORDER)}
    pos_idx = {v: i for i, v in enumerate(POS_ORDER)}

    return fa_map, fa_idx, pos_map, pos_idx


def patch_latents(lat_csv: Path, fa_map, fa_idx, pos_map, pos_idx) -> int:
    df = pd.read_csv(lat_csv)
    df["_key"] = df["filename"].apply(lambda p: Path(p).stem.replace("_", "-"))

    df["annotation_label"]      = df["_key"].map(fa_map).map(fa_idx).fillna(-1).astype(int)
    df["annotation_label_name"] = df["_key"].map(fa_map).fillna("")
    df["annotation_label_name"] = df.apply(
        lambda r: r["annotation_label_name"] if r["annotation_label"] != -1 else "", axis=1)

    df["annotation_label_2"]      = df["_key"].map(pos_map).map(pos_idx).fillna(-1).astype(int)
    df["annotation_label_2_name"] = df["_key"].map(pos_map).fillna("")
    df["annotation_label_2_name"] = df.apply(
        lambda r: r["annotation_label_2_name"] if r["annotation_label_2"] != -1 else "", axis=1)

    df = df.drop(columns=["_key"])
    df.to_csv(lat_csv, index=False)

    n = (df["annotation_label"] != -1).sum()
    return int(n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--label-csv", type=Path, default=Path(LABEL_CSV_DEFAULT))
    parser.add_argument("--boxplot-kind", default="box", choices=["box", "violin"])
    args = parser.parse_args()

    if not args.run_dir.is_dir():
        sys.exit(f"Not a directory: {args.run_dir}")
    if not args.label_csv.exists():
        sys.exit(f"Label CSV not found: {args.label_csv}")

    fa_map, fa_idx, pos_map, pos_idx = _build_label_maps(args.label_csv)

    for variant_dir in sorted(args.run_dir.iterdir()):
        if not variant_dir.is_dir():
            continue
        lat_csv = variant_dir / "latents.csv"
        if not lat_csv.exists():
            print(f"  [skip] no latents.csv in {variant_dir.name}")
            continue

        n = patch_latents(lat_csv, fa_map, fa_idx, pos_map, pos_idx)
        print(f"  {variant_dir.name}: annotated {n} patches in latents.csv")

        recon_dir = variant_dir / "recon"
        if not (recon_dir / "patches_raw.tif").exists():
            print(f"  [skip plots] no patches_raw.tif in {variant_dir.name}")
            continue

        process_variant(variant_dir, args.boxplot_kind)


if __name__ == "__main__":
    main()
