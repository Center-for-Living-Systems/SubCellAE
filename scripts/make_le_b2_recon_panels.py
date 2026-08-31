#!/usr/bin/env python3
"""
make_le_b2_recon_panels.py

For each completed SupCon-AE run, produce a reconstruction panel showing
the model's input vs. reconstruction quality across all 5 FA sub-classes.

Layout: 10 rows × 10 columns
  Row 1 (odd)  : No Adhesion          — original patches
  Row 2 (even) : No Adhesion          — reconstructions
  Row 3 (odd)  : Nascent Adhesion     — original
  Row 4 (even) : Nascent Adhesion     — reconstruction
  Row 5 (odd)  : Focal Complex        — original
  Row 6 (even) : Focal Complex        — reconstruction
  Row 7 (odd)  : Focal Adhesion       — original
  Row 8 (even) : Focal Adhesion       — reconstruction
  Row 9 (odd)  : Fibrillar Adhesion   — original
  Row 10 (even): Fibrillar Adhesion   — reconstruction

The same 50 patches (10 per class, fixed seed=42) are used for every run
so quality differences across budgets are directly comparable.
Border colour = FA sub-class.
Output: {result_dir}/recon_panel.png

Usage
-----
  # All completed DS1 runs:
  python scripts/make_le_b2_recon_panels.py --dataset ds1

  # Single run:
  python scripts/make_le_b2_recon_panels.py --run le_b2_ds1_fv0_nb100_r0

  # Skip runs whose panel already exists:
  python scripts/make_le_b2_recon_panels.py --dataset ds1 --skip-existing
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import tifffile
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO     = Path(__file__).resolve().parents[1]
DATA     = Path("/net/projects/CLS/lding/data/fa_data_analysis")
RUN_ROOT = DATA / "ae_results/contrastive_run/le_b2_supcon"
PATCH_ROOT = DATA / "ae_results/patches/cio"

ANN_FILES_5CLS = {
    "ds1": DATA / "labelling/vinc_combined_label_Annabel_20260816.csv",
    "ds2": DATA / "labelling/pfak_combined_label_Annabel_aug2026.csv",
    "ds3": DATA / "labelling/ppax_control_label_Ernest_20260825_1433.csv",
}

PATCH_DIRS = {
    "ds1": [PATCH_ROOT / "vinc/control/tiff_patches32_mr10",
            PATCH_ROOT / "vinc/ycomp/tiff_patches32_mr10"],
    "ds2": [PATCH_ROOT / "pfak/control/tiff_patches32_mr10",
            PATCH_ROOT / "pfak/ycomp/tiff_patches32_mr10"],
    "ds3": [PATCH_ROOT / "ppax/control/tiff_patches32_mr10",
            PATCH_ROOT / "ppax/ycomp/tiff_patches32_mr10"],
}

FA5_ORDER = [
    "No adhesion",
    "Nascent Adhesion",
    "focal complex",
    "focal adhesion",
    "fibrillar adhesion",
]
FA5_COLORS = {
    "No adhesion":        "#9467bd",
    "Nascent Adhesion":   "#1f77b4",
    "focal complex":      "#ff7f0e",
    "focal adhesion":     "#2ca02c",
    "fibrillar adhesion": "#d62728",
}
FA5_SHORT = {
    "No adhesion":        "No Adh",
    "Nascent Adhesion":   "Nascent",
    "focal complex":      "FC",
    "focal adhesion":     "FA",
    "fibrillar adhesion": "Fib",
}

N_PER_CLASS  = 10
SAMPLE_SEED  = 42
BORDER_PX    = 1
INPUT_DIVISOR = 2.0   # matches enlarged_crop.input_divisor in training configs


# ---------------------------------------------------------------------------
# Helpers

def _norm01(img: np.ndarray) -> np.ndarray:
    lo, hi = img.min(), img.max()
    return (img - lo) / (hi - lo + 1e-9)


def _hex_to_rgb01(h: str):
    return (int(h[1:3], 16) / 255, int(h[3:5], 16) / 255, int(h[5:7], 16) / 255)


def _add_border(img: np.ndarray, color_hex: str, px: int = BORDER_PX) -> np.ndarray:
    """img: (H, W) float in [0,1] → (H+2px, W+2px, 3) RGB."""
    r, g, b = _hex_to_rgb01(color_hex)
    h, w    = img.shape
    out     = np.ones((h + 2 * px, w + 2 * px, 3), dtype=np.float32)
    out[:, :] = [r, g, b]
    out[px:px + h, px:px + w] = np.stack([img, img, img], axis=-1)
    return out


def _find_patch(filename: str, patch_dirs: list[Path]) -> Path | None:
    for d in patch_dirs:
        p = d / filename
        if p.exists():
            return p
    return None


def _load_patch(path: Path, input_divisor: float = INPUT_DIVISOR) -> np.ndarray:
    """Load TIFF, divide by input_divisor (matching training), return (H,W) float32."""
    arr = tifffile.imread(str(path)).astype(np.float32)
    return arr / input_divisor


def _sample_patches(ds: str) -> dict[str, list[str]]:
    """Return {class: [filename, ...]} — same selection for every run."""
    df  = pd.read_csv(ANN_FILES_5CLS[ds])
    rng = np.random.default_rng(SAMPLE_SEED)
    sel = {}
    for cls in FA5_ORDER:
        pool = df[df["label"] == cls]["filename"].tolist()
        n    = min(N_PER_CLASS, len(pool))
        sel[cls] = list(rng.choice(pool, n, replace=False))
    return sel


def _run_inference(model, patches_raw: list[np.ndarray], device: str) -> list[np.ndarray]:
    """Forward pass on a list of (H,W) float32 arrays. Returns list of (H,W) recon arrays."""
    model.eval()
    recons = []
    with torch.no_grad():
        for arr in patches_raw:
            x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            recon, _ = model(x)
            recons.append(recon.squeeze().cpu().numpy())
    return recons


# ---------------------------------------------------------------------------
# Panel builder

def make_panel(run_dir: Path, ds: str,
               patch_sel: dict[str, list[str]],
               patch_dirs: list[Path],
               device: str = "cpu") -> Path | None:
    """Build and save recon_panel.png inside run_dir. Returns path or None if skipped."""
    model_path = run_dir / "model_final.pt"
    if not model_path.exists():
        return None

    model = torch.load(str(model_path), map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()

    n_cols   = N_PER_CLASS
    n_rows   = len(FA5_ORDER) * 2   # orig + recon per class
    cell_px  = 32 + 2 * BORDER_PX   # 32 patch + border

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 0.5, n_rows * 0.6),
        facecolor="black",
    )

    for ci_cls, cls in enumerate(FA5_ORDER):
        row_orig  = ci_cls * 2        # 0, 2, 4, 6, 8
        row_recon = ci_cls * 2 + 1    # 1, 3, 5, 7, 9
        color     = FA5_COLORS[cls]
        filenames = patch_sel[cls]

        # Load patches and run inference
        raws = []
        for fn in filenames:
            p = _find_patch(fn, patch_dirs)
            if p is not None:
                raws.append(_load_patch(p))
            else:
                raws.append(np.zeros((32, 32), dtype=np.float32))

        recons = _run_inference(model, raws, device)

        for col_i, (raw, rec) in enumerate(zip(raws, recons)):
            if col_i >= n_cols:
                break

            orig_disp  = _add_border(_norm01(raw),  color)
            recon_disp = _add_border(_norm01(rec),  color)

            ax_o = axes[row_orig,  col_i]
            ax_r = axes[row_recon, col_i]

            ax_o.imshow(orig_disp,  vmin=0, vmax=1, interpolation="nearest")
            ax_r.imshow(recon_disp, vmin=0, vmax=1, interpolation="nearest")

            for ax in (ax_o, ax_r):
                ax.axis("off")

        # Row labels on leftmost column
        r_hex  = _hex_to_rgb01(color)
        for row, suffix in [(row_orig, "orig"), (row_recon, "recon")]:
            axes[row, 0].set_ylabel(
                f"{FA5_SHORT[cls]}\n{suffix}",
                fontsize=5.5, color=r_hex, rotation=0,
                labelpad=28, va="center",
            )
            axes[row, 0].axis("on")
            axes[row, 0].set_xticks([])
            axes[row, 0].set_yticks([])
            for spine in axes[row, 0].spines.values():
                spine.set_visible(False)

    run_name = run_dir.name
    fig.suptitle(run_name, fontsize=6, color="white", y=1.01)
    fig.tight_layout(pad=0.1)

    out = run_dir / "recon_panel.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Main

def main():
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--dataset", choices=["ds1", "ds2", "ds3"],
                     help="Process all completed runs for this dataset")
    grp.add_argument("--run", help="Process a single run by name (e.g. le_b2_ds1_fv0_nb100_r0)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip runs that already have recon_panel.png")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = args.device

    if args.run:
        run_dir = RUN_ROOT / args.run
        ds      = args.run.split("_")[2]   # e.g. "ds1"
        runs    = [(run_dir, ds)]
    else:
        ds   = args.dataset
        dirs = sorted(RUN_ROOT.glob(f"le_b2_{ds}_*"))
        runs = [(d, ds) for d in dirs if d.is_dir()]

    # Pre-compute fixed patch selection per dataset
    patch_sel_cache: dict[str, dict] = {}

    for run_dir, ds in runs:
        out_path = run_dir / "recon_panel.png"
        if args.skip_existing and out_path.exists():
            print(f"  skip (exists): {run_dir.name}")
            continue
        if not (run_dir / "model_final.pt").exists():
            print(f"  skip (no model): {run_dir.name}")
            continue

        if ds not in patch_sel_cache:
            patch_sel_cache[ds] = _sample_patches(ds)

        print(f"  processing: {run_dir.name} ...", end="", flush=True)
        out = make_panel(
            run_dir, ds,
            patch_sel=patch_sel_cache[ds],
            patch_dirs=PATCH_DIRS[ds],
            device=device,
        )
        if out:
            print(f" → {out.name}")
        else:
            print(" → skipped (no model)")

    print("Done.")


if __name__ == "__main__":
    main()
