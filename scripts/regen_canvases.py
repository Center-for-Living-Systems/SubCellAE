#!/usr/bin/env python3
"""Regenerate reconstruction canvas images for models trained on multiple datasets.

Reads existing patches_raw.tif / patches_recon.tif + patches_index.csv from
<variant_dir>/recon/ and rebuilds per-source-image canvases, namespacing group
keys by dataset so vinc control_f0001 and ppax control_f0001 are separate.

Output format — all grayscale float32 (ImageJ contrast-adjustable):
  visual.tif        (N, title_h + n_ch*H, 2W)  stack; left=raw, right=recon
  visual_index.csv  frame, group
  visual_{group}.tif  individual grayscale files, same layout
  images_raw.tif / images_recon.tif  unchanged (H,W) per-channel stacks

Usage:
    python scripts/regen_canvases.py <variant_dir> [<variant_dir2> ...]
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image, ImageDraw, ImageFont

_COORD_RE = re.compile(r'^(.+_f\d+)x(\d+)y(\d+)ps(\d+)$')

RECON_PAD_SIZE   = 64
RECON_IMAGE_SIZE = 1024
TITLE_HEIGHT     = 28   # px for title bar


def _parse_coords(name: str):
    m = _COORD_RE.match(name)
    if m is None:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))


def _get_ch(arr, ch: int):
    return arr[ch] if arr.ndim == 3 else arr


def _add_title(canvas: np.ndarray, title: str, title_h: int = TITLE_HEIGHT) -> np.ndarray:
    """Prepend a dark title bar with white text to a (H, W) float32 array."""
    W = canvas.shape[1]
    bar = np.full((title_h, W), 0.12, dtype=np.float32)   # dark gray

    # draw text using PIL on an 8-bit image, then rescale
    pil = Image.fromarray((bar * 255).astype(np.uint8), mode="L")
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    draw.text((6, 5), title, fill=230, font=font)
    bar = np.array(pil).astype(np.float32) / 255.0

    return np.concatenate([bar, canvas], axis=0)


def regen_one(variant_dir: Path, pad: int, img_size: int):
    recon_dir  = variant_dir / "recon"
    idx_path   = recon_dir / "patches_index.csv"
    raw_path   = recon_dir / "patches_raw.tif"
    recon_path = recon_dir / "patches_recon.tif"

    for p in [idx_path, raw_path, recon_path]:
        if not p.exists():
            sys.exit(f"Missing: {p}")

    print(f"\n{'='*60}")
    print(f"Regenerating: {variant_dir.name}")
    idx_df      = pd.read_csv(idx_path)
    raw_stack   = tifffile.imread(str(raw_path))
    recon_stack = tifffile.imread(str(recon_path))
    print(f"  Patches: {len(idx_df)}   raw_stack: {raw_stack.shape}")

    n_ch = raw_stack.shape[1] if raw_stack.ndim == 4 else 1

    # canvas_data[ns_group][split] = [(r0,r1,c0,c1, raw_p, recon_p)]
    canvas_data = defaultdict(lambda: defaultdict(list))

    for _, row in idx_df.iterrows():
        frame  = int(row["frame"])
        split  = str(row.get("split", "train"))
        cname  = str(row.get("condition_name", ""))
        name   = str(row["name"])

        raw_p   = raw_stack[frame]
        recon_p = recon_stack[frame]

        coords = _parse_coords(name)
        if coords is None:
            continue
        frame_group, x_c, y_c, ps = coords

        dataset_prefix = cname.split("_")[0] if "_" in cname else cname
        ns_group = f"{dataset_prefix}_{frame_group}" if dataset_prefix else frame_group

        half = ps // 2
        r0 = y_c - half - pad;  r1 = y_c + half - pad
        c0 = x_c - half - pad;  c1 = x_c + half - pad
        if r0 < 0 or c0 < 0:
            continue

        canvas_data[ns_group][split].append((r0, r1, c0, c1, raw_p, recon_p))

    all_groups = sorted(canvas_data.keys())
    print(f"  Canvas groups: {len(all_groups)}")

    img_stack_raw:   list = []
    img_stack_recon: list = []
    img_stack_index: list = []
    vis_stack:       list = []
    vis_index:       list = []

    for group in all_groups:
        split_entries  = canvas_data[group]
        splits_present = sorted(split_entries.keys())
        all_entries    = [(r0, r1, c0, c1, rp, rcp, sp)
                          for sp, entries in split_entries.items()
                          for (r0, r1, c0, c1, rp, rcp) in entries]
        if not all_entries:
            continue

        raw_canvases   = [np.zeros((img_size, img_size), dtype=np.float32) for _ in range(n_ch)]
        recon_canvases = [np.zeros((img_size, img_size), dtype=np.float32) for _ in range(n_ch)]

        for r0, r1, c0, c1, rp, rcp, _sp in all_entries:
            r1 = min(r1, img_size);  c1 = min(c1, img_size)
            for ch in range(n_ch):
                raw_canvases[ch][r0:r1, c0:c1]  = _get_ch(rp,  ch)[:r1-r0, :c1-c0]
                recon_canvases[ch][r0:r1, c0:c1] = _get_ch(rcp, ch)[:r1-r0, :c1-c0]

        # images_raw / images_recon stacks
        for ch in range(n_ch):
            img_stack_index.append({"frame": len(img_stack_raw), "group": group, "channel": ch})
            img_stack_raw.append(raw_canvases[ch])
            img_stack_recon.append(recon_canvases[ch])

        # visual frame: (n_ch * H, 2W) side-by-side, with title bar on top
        title = f"{group}  [{'+'.join(splits_present)}]"
        rows = [np.concatenate([raw_canvases[ch], recon_canvases[ch]], axis=1)
                for ch in range(n_ch)]
        canvas_2w = np.concatenate(rows, axis=0)        # (n_ch*H, 2W)
        vis_arr   = _add_title(canvas_2w, title)         # (title_h + n_ch*H, 2W)

        # individual grayscale TIFF
        tifffile.imwrite(str(recon_dir / f"visual_{group}.tif"), vis_arr)

        vis_index.append({"frame": len(vis_stack), "group": group})
        vis_stack.append(vis_arr)

    # write stacked outputs
    if img_stack_raw:
        tifffile.imwrite(str(recon_dir / "images_raw.tif"),
                         np.stack(img_stack_raw,   axis=0), imagej=True)
        tifffile.imwrite(str(recon_dir / "images_recon.tif"),
                         np.stack(img_stack_recon, axis=0), imagej=True)
        pd.DataFrame(img_stack_index).to_csv(recon_dir / "images_index.csv", index=False)
        print(f"  images_raw/recon.tif  ({len(img_stack_raw)} frames)")

    if vis_stack:
        tifffile.imwrite(str(recon_dir / "visual.tif"),
                         np.stack(vis_stack, axis=0))
        pd.DataFrame(vis_index).to_csv(recon_dir / "visual_index.csv", index=False)
        print(f"  visual.tif            ({len(vis_stack)} frames)")

    print(f"  Done → {recon_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant_dirs", nargs="+", type=Path)
    parser.add_argument("--pad",      type=int, default=RECON_PAD_SIZE)
    parser.add_argument("--img-size", type=int, default=RECON_IMAGE_SIZE)
    args = parser.parse_args()
    for vdir in args.variant_dirs:
        if not vdir.is_dir():
            print(f"Skipping (not a dir): {vdir}"); continue
        regen_one(vdir, args.pad, args.img_size)


if __name__ == "__main__":
    main()
