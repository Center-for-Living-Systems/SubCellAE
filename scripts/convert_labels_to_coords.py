"""
convert_labels_to_coords.py
===========================
Convert patch-filename-based label CSVs into coordinate-based CSVs.

The current label CSVs identify patches by filename, e.g.:
    control_f0001x0592y0560ps32.tif   (or unique_ID: control-f0001x0592y0560ps32.tif)

Those filenames encode *padded* coordinates (cx_pad = cx + pad_size, same for cy).
This script:
  1. Parses (condition, frame_idx, cx_pad, cy_pad, ps) from filenames
  2. Recovers frame-space center: cx = cx_pad - pad_size
  3. Saves a coord CSV usable with any patch size via CoordCropDataset

Usage
-----
Convert the full unlabelled patch pool from a tiff_patches directory:
    python scripts/convert_labels_to_coords.py \\
        --patch_dir  <path>/tiff_patches32_mr10 \\
        --dataset    vinc \\
        --condition_name control \\
        --out        <path>/labelling/vinc_control_coords.csv

Convert and merge with a B1/B2 label file:
    python scripts/convert_labels_to_coords.py \\
        --patch_dir  <path>/tiff_patches32_mr10 \\
        --dataset    vinc \\
        --condition_name control \\
        --label_csv  <path>/labelling/vinc_b1_matched_ds1.csv \\
        --label_col  label \\
        --out        <path>/labelling/vinc_b1_coords.csv

Merge B1 + B2 into a single coord CSV (add --append):
    python scripts/convert_labels_to_coords.py \\
        --patch_dir  <path>/tiff_patches32_mr10 \\
        --dataset    vinc \\
        --condition_name control \\
        --label_csv  <path>/labelling/vinc_b2_matched_ds1.csv \\
        --label_col  label \\
        --out        <path>/labelling/vinc_b1_coords.csv \\   # existing file
        --append

Output columns
--------------
    dataset         : str   – dataset name (e.g. "vinc")
    condition       : int   – condition id (0=control, 1=ycomp)
    condition_name  : str   – human-readable condition ("control", "ycomp")
    frame_idx       : int   – 0-based frame index
    cx              : int   – center x in frame coordinates (unpadded)
    cy              : int   – center y in frame coordinates (unpadded)
    source_ps       : int   – patch size encoded in original filename (32)
    split           : str   – "train" or "val" (frame-group-aware, 75/25)
    label           : str   – annotation label (empty if unlabelled)
    annotator       : str   – annotator name (from label CSV if present)
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Filename pattern: {condition}_f{frame}x{cx_pad}y{cy_pad}ps{ps}.tif
_COORD_RE = re.compile(
    r'^(.+)_f(\d+)x(\d+)y(\d+)ps(\d+)\.(tiff?)$', re.IGNORECASE
)
# unique_ID uses hyphens: control-f0001x…
_COORD_RE_HYPHEN = re.compile(
    r'^(.+)-f(\d+)x(\d+)y(\d+)ps(\d+)\.(tiff?)$', re.IGNORECASE
)

PAD_SIZE = 64   # patchprep pads coordinates by this amount


def _parse_filename(fn: str) -> dict | None:
    """Return parsed coordinate dict from patch filename, or None."""
    name = Path(fn).name
    m = _COORD_RE.match(name) or _COORD_RE_HYPHEN.match(name)
    if not m:
        return None
    cond_str, frame_idx, cx_pad, cy_pad, ps, _ = m.groups()
    return {
        "condition_name": cond_str,
        "frame_idx":      int(frame_idx),
        "cx":             int(cx_pad) - PAD_SIZE,
        "cy":             int(cy_pad) - PAD_SIZE,
        "source_ps":      int(ps),
    }


def _build_pool_from_patch_dir(patch_dir: Path, dataset: str, condition: int) -> pd.DataFrame:
    """Build a coord DataFrame from all TIF filenames in patch_dir."""
    rows = []
    for fn in sorted(patch_dir.iterdir()):
        if fn.suffix.lower() not in {".tif", ".tiff"}:
            continue
        parsed = _parse_filename(fn.name)
        if parsed is None:
            print(f"  Warning: skipping unrecognised filename {fn.name}")
            continue
        rows.append({
            "dataset":        dataset,
            "condition":      condition,
            "condition_name": parsed["condition_name"],
            "frame_idx":      parsed["frame_idx"],
            "cx":             parsed["cx"],
            "cy":             parsed["cy"],
            "source_ps":      parsed["source_ps"],
        })
    df = pd.DataFrame(rows)
    print(f"  Pool: {len(df)} patches from {patch_dir.name}")
    return df


def _assign_splits(df: pd.DataFrame, val_frac: float = 0.25,
                   random_state: int = 42) -> pd.DataFrame:
    """Assign train/val split by frame group (all patches from one frame → one split)."""
    rng = np.random.default_rng(random_state)
    frames = sorted(df["frame_idx"].unique())
    n_val = max(1, round(len(frames) * val_frac))
    val_frames = set(rng.choice(frames, size=n_val, replace=False).tolist())
    df = df.copy()
    df["split"] = df["frame_idx"].apply(lambda f: "val" if f in val_frames else "train")
    return df


def _merge_labels(pool: pd.DataFrame, label_csv: Path,
                  label_col: str, filename_col: str) -> pd.DataFrame:
    """Merge annotation labels from label_csv into the pool DataFrame."""
    ldf = pd.read_csv(label_csv)
    ldf = ldf[[filename_col] + [c for c in ldf.columns if c != filename_col]]

    # Parse coordinates from the label filename column
    label_rows = []
    for _, row in ldf.iterrows():
        parsed = _parse_filename(str(row[filename_col]))
        if parsed is None:
            continue
        label_rows.append({
            "condition_name": parsed["condition_name"],
            "frame_idx":      parsed["frame_idx"],
            "cx":             parsed["cx"],
            "cy":             parsed["cy"],
            "label":          str(row[label_col]),
            "annotator":      str(row.get("annotator", "")),
        })
    ldf_coord = pd.DataFrame(label_rows)

    # Merge on (condition_name, frame_idx, cx, cy)
    join_keys = ["condition_name", "frame_idx", "cx", "cy"]
    merged = pool.merge(
        ldf_coord[join_keys + ["label", "annotator"]].drop_duplicates(join_keys),
        on=join_keys, how="left",
    )
    n_matched = merged["label"].notna().sum()
    print(f"  Merged labels: {n_matched} / {len(merged)} matched from {label_csv.name}")
    return merged


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patch_dir",       required=True,  help="Directory of pre-saved patch TIFs (for coordinate enumeration)")
    p.add_argument("--dataset",         required=True,  help="Dataset name, e.g. 'vinc'")
    p.add_argument("--condition",       default=0,      type=int, help="Condition integer ID (default: 0)")
    p.add_argument("--condition_name",  default="",     help="Override condition name (default: parsed from filename)")
    p.add_argument("--label_csv",       default=None,   help="Optional label CSV to merge in")
    p.add_argument("--label_col",       default="label", help="Label column in label_csv (default: 'label')")
    p.add_argument("--filename_col",    default="unique_ID", help="Filename column in label_csv (default: 'unique_ID')")
    p.add_argument("--pad_size",        default=64,     type=int, help="Pad size used in patchprep (default: 64)")
    p.add_argument("--val_frac",        default=0.25,   type=float, help="Validation fraction for frame-split (default: 0.25)")
    p.add_argument("--out",             required=True,  help="Output CSV path")
    p.add_argument("--append",          action="store_true", help="Append to existing output CSV (for merging B1+B2)")
    args = p.parse_args()

    global PAD_SIZE
    PAD_SIZE = args.pad_size

    patch_dir = Path(args.patch_dir)
    if not patch_dir.is_dir():
        raise FileNotFoundError(f"patch_dir not found: {patch_dir}")

    print(f"Building coord pool from {patch_dir} …")
    pool = _build_pool_from_patch_dir(patch_dir, args.dataset, args.condition)

    if args.condition_name:
        pool["condition_name"] = args.condition_name

    pool = _assign_splits(pool, args.val_frac)

    if args.label_csv:
        label_csv = Path(args.label_csv)
        print(f"Merging labels from {label_csv.name} …")
        pool = _merge_labels(pool, label_csv, args.label_col, args.filename_col)
    else:
        pool["label"]     = ""
        pool["annotator"] = ""

    # Column order
    cols = ["dataset", "condition", "condition_name", "frame_idx",
            "cx", "cy", "source_ps", "split", "label", "annotator"]
    pool = pool[cols]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.append and out_path.exists():
        existing = pd.read_csv(out_path)
        pool = pd.concat([existing, pool], ignore_index=True)
        # De-duplicate by (dataset, condition_name, frame_idx, cx, cy) keeping last (new labels win)
        key = ["dataset", "condition_name", "frame_idx", "cx", "cy"]
        pool = pool.drop_duplicates(key, keep="last")
        print(f"  Appended; total rows: {len(pool)}")

    pool.to_csv(out_path, index=False)
    print(f"Saved → {out_path}  ({len(pool)} rows)")
    print(pool["label"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
