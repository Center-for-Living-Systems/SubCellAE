#!/usr/bin/env python3
"""
analyze_fa4_errors.py
=====================
Inspect misclassified patches from the FA4 cross-dataset label-efficiency experiments.

For each scenario, trains a LightGBM at 75% labels (first repeat) and collects
per-patch predictions on the test set.  Saves:
  - predictions_{scenario}_{suffix}.csv  (filename, dataset, true_label, pred_label)
  - errors_{scenario}_{suffix}.png       (grid of misclassified patches, 5 per cell)

Usage
-----
  python scripts/analyze_fa4_errors.py                        # Option A, zrecon, 75% frac
  python scripts/analyze_fa4_errors.py --suffix A_zproj
  python scripts/analyze_fa4_errors.py --scenario vinc_only
  python scripts/analyze_fa4_errors.py --frac 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

# ---------------------------------------------------------------------------
DATA_ROOT = Path("/net/projects/CLS/lding/data/fa_data_analysis")
EVAL_DIR  = DATA_ROOT / "ae_results" / "contrastive_run" / "fa4_xds_eval"
LABEL_DIR = DATA_ROOT / "labelling"

PATCH_DIRS = {
    "vinc_ctrl":  DATA_ROOT / "ae_results" / "patches" / "cio"    / "vinc" / "control" / "tiff_patches32_mr10",
    "vinc_ycomp": DATA_ROOT / "ae_results" / "patches" / "cio"    / "vinc" / "ycomp"   / "tiff_patches32_mr10",
    "pfak_ctrl":  DATA_ROOT / "ae_results" / "patches" / "cio_rb" / "pfak" / "control" / "tiff_patches32_mr10",
}

LABEL_FILES = {
    "vinc_ctrl":  LABEL_DIR / "vinc_control_label_Annabel_20260715_1554.csv",
    "vinc_ycomp": LABEL_DIR / "vinc_combined_label_Annabel_20260816.csv",
    "pfak_ctrl":  LABEL_DIR / "pfak_combined_label_Annabel_aug2026.csv",
}

FA_LABEL_ORDER_4 = [
    "Nascent Adhesion",
    "focal complex",
    "focal adhesion",
    "fibrillar adhesion",
]
FA_LABEL_ORDER_5 = [
    "No adhesion",
    "Nascent Adhesion",
    "focal complex",
    "focal adhesion",
    "fibrillar adhesion",
]
LABEL_SHORT = {
    "No adhesion":        "NoAd",
    "Nascent Adhesion":   "NA",
    "focal complex":      "FC",
    "focal adhesion":     "FA",
    "fibrillar adhesion": "Fib",
}
LABEL_COLORS = {
    "No adhesion":        "#9467bd",  # purple
    "Nascent Adhesion":   "#1565C0",  # blue
    "focal complex":      "#E65100",  # orange
    "focal adhesion":     "#2ca02c",  # green
    "fibrillar adhesion": "#C00000",  # red
}
SEED = 42

Z_COLS  = [f"z_{i}"  for i in range(12)]
ZP_COLS = [f"zp_{i}" for i in range(8)]

SCENARIOS = [
    ("vinc_only",   ["vinc_ctrl", "vinc_ycomp"], ["vinc_ctrl", "vinc_ycomp"], False),
    ("pfak_only",   ["pfak_ctrl"],               ["pfak_ctrl"],               False),
    ("ctrl->ycomp", ["vinc_ctrl"],               ["vinc_ycomp"],              True),
    ("vinc->pfak",  ["vinc_ctrl", "vinc_ycomp"], ["pfak_ctrl"],               True),
    ("pfak->vinc",  ["pfak_ctrl"],               ["vinc_ctrl", "vinc_ycomp"], True),
    ("combined",    ["vinc_ctrl", "vinc_ycomp", "pfak_ctrl"],
                    ["vinc_ctrl", "vinc_ycomp", "pfak_ctrl"],                 False),
]


def _feat_cols(variant: str) -> list[str]:
    if variant == "zproj": return ZP_COLS
    if variant == "both":  return Z_COLS + ZP_COLS
    return Z_COLS


def _load_labels(ds_key: str, label_order: list[str] | None = None) -> pd.DataFrame:
    if label_order is None:
        label_order = FA_LABEL_ORDER_4
    p = LABEL_FILES[ds_key]
    df = pd.read_csv(p)
    df = df[df["label"].isin(label_order)][["filename", "label"]].copy()
    return df


def _load_latents(ds_key: str) -> pd.DataFrame:
    csv_map = {
        "vinc_ctrl":  EVAL_DIR / "encoded_vinc_ctrl.csv",
        "vinc_ycomp": EVAL_DIR / "encoded_vinc_ycomp.csv",
        "pfak_ctrl":  EVAL_DIR / "encoded_pfak_ctrl.csv",
    }
    return pd.read_csv(csv_map[ds_key])


def _build_dataset(ds_keys: list[str], label_order: list[str] | None = None) -> pd.DataFrame:
    if label_order is None:
        label_order = FA_LABEL_ORDER_4
    frames = []
    for k in ds_keys:
        lat  = _load_latents(k)
        labs = _load_labels(k, label_order)
        merged = lat.merge(labs, on="filename", how="inner")
        merged["dataset"] = k
        frames.append(merged)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["label"].isin(label_order)].copy()
    df["label"] = pd.Categorical(df["label"], categories=label_order, ordered=True)
    return df


def make_classifier():
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            min_child_samples=3, class_weight="balanced",
            random_state=SEED, verbose=-1, n_jobs=1,
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(n_estimators=300, random_state=SEED)


BORDER_PX = 2


def _hex_to_rgb01(hex_str: str) -> tuple[float, float, float]:
    h = hex_str.lstrip("#")
    return int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255


def _to_gray(img: np.ndarray) -> np.ndarray:
    """Reduce to 2-D grayscale regardless of input shape."""
    if img.ndim == 2:
        return img
    if img.shape[0] <= 4:   # C×H×W
        return img[0]
    return img[:, :, 0]     # H×W×C


def _add_border(img_2d: np.ndarray, color_hex: str, border: int = BORDER_PX) -> np.ndarray:
    """Add a solid-color border and convert grayscale → RGB float32."""
    h, w = img_2d.shape
    r, g, b = _hex_to_rgb01(color_hex)
    out = np.empty((h + 2 * border, w + 2 * border, 3), dtype=np.float32)
    out[:, :, 0] = r
    out[:, :, 1] = g
    out[:, :, 2] = b
    gray = np.stack([img_2d, img_2d, img_2d], axis=-1)
    out[border:border + h, border:border + w, :] = gray
    return out


def _load_patch(filename: str, ds_key: str) -> np.ndarray | None:
    """Load 32×32 patch TIFF; return None if not found."""
    path = PATCH_DIRS[ds_key] / filename
    if not path.exists():
        return None
    img = tifffile.imread(str(path)).astype(np.float32)
    return img


def _norm(img: np.ndarray) -> np.ndarray:
    lo, hi = img.min(), img.max()
    if hi > lo:
        return (img - lo) / (hi - lo)
    return np.zeros_like(img)


def _plot_errors(df_errors: pd.DataFrame, scenario_name: str, suffix: str,
                 n_per_cell: int = 5, label_order: list[str] | None = None):
    """Grid of misclassified patches: rows=true class, cols=predicted class.

    Each patch is shown at 1:1 aspect ratio with a colored bounding box
    matching the true class color.
    """
    if label_order is None:
        label_order = FA_LABEL_ORDER_4
    classes = label_order
    n_cls   = len(classes)

    grid: dict[tuple, list] = {}
    for cls_t in classes:
        for cls_p in classes:
            if cls_t == cls_p:
                continue
            cell = df_errors[
                (df_errors["true_label"] == cls_t) &
                (df_errors["pred_label"] == cls_p)
            ]
            if len(cell) == 0:
                continue
            grid[(cls_t, cls_p)] = cell.head(n_per_cell)

    if not grid:
        print(f"[{scenario_name}] No misclassified patches to display.")
        return

    # Cell size: n_per_cell patches side-by-side at 1:1 aspect → width = n_per_cell × height
    unit = 0.55   # inches per patch height
    fig_w = unit * n_per_cell * n_cls + 1.0
    fig_h = unit * n_cls + 1.2
    fig, outer = plt.subplots(n_cls, n_cls, figsize=(fig_w, fig_h),
                               gridspec_kw={"wspace": 0.06, "hspace": 0.22})

    empty_patch = np.zeros((32, 32), dtype=np.float32)

    for ri, cls_t in enumerate(classes):
        for ci, cls_p in enumerate(classes):
            ax = outer[ri][ci] if n_cls > 1 else outer
            ax.axis("off")

            if ri == 0:
                ax.set_title(f"pred: {LABEL_SHORT[cls_p]}", fontsize=7,
                             color=LABEL_COLORS[cls_p], fontweight="bold", pad=2)
            if ci == 0:
                ax.set_ylabel(f"true: {LABEL_SHORT[cls_t]}", fontsize=7,
                              color=LABEL_COLORS[cls_t], fontweight="bold")
                ax.yaxis.set_label_position("left")
                ax.tick_params(left=False, labelleft=False)

            if ri == ci:
                ax.set_facecolor("#f5f5f5")
                ax.text(0.5, 0.5, "✓", ha="center", va="center",
                        transform=ax.transAxes, fontsize=14, color="gray")
                continue

            cell_df = grid.get((cls_t, cls_p))
            if cell_df is None or len(cell_df) == 0:
                ax.text(0.5, 0.5, "0 errors", ha="center", va="center",
                        transform=ax.transAxes, fontsize=6, color="lightgray")
                continue

            # Build bordered patches (RGB, 1:1 per patch)
            true_color = LABEL_COLORS[cls_t]
            bordered = []
            for _, row in cell_df.iterrows():
                img = _load_patch(row["filename"], row["dataset"])
                if img is not None:
                    gray = _norm(_to_gray(img))
                    bordered.append(_add_border(gray, true_color))

            if not bordered:
                continue

            # Pad with dark empty patches if fewer than n_per_cell loaded
            while len(bordered) < n_per_cell:
                bordered.append(_add_border(empty_patch, true_color))

            strip = np.concatenate(bordered[:n_per_cell], axis=1)  # H×(n*W)×3
            ax.imshow(strip, aspect="equal", interpolation="nearest")
            n_err = len(cell_df)
            ax.text(0.5, 1.01, f"n={n_err}", ha="center", va="bottom",
                    transform=ax.transAxes, fontsize=6, color="red")

    total_err = len(df_errors)
    total_test = len(df_errors)  # only errors passed in
    fig.suptitle(
        f"Misclassified patches — {scenario_name}  ({suffix})\n"
        f"{total_err} total errors  |  showing up to {n_per_cell} per cell",
        fontsize=11, fontweight="bold",
    )
    out = EVAL_DIR / f"errors_{scenario_name}_{suffix}.png"
    fig.savefig(str(out), dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {out}")


def run_scenario(scenario_name: str, train_keys: list[str], test_keys: list[str],
                 cross_ds: bool, fcols: list[str], frac: float, suffix: str,
                 label_order: list[str] | None = None):
    from sklearn.model_selection import StratifiedShuffleSplit

    if label_order is None:
        label_order = FA_LABEL_ORDER_4

    print(f"\n[{scenario_name}] train={train_keys}  test={test_keys}  frac={frac}")

    all_keys = set(train_keys + test_keys)
    ds_data = {k: _build_dataset([k], label_order) for k in all_keys}

    train_pool = pd.concat([ds_data[k] for k in train_keys], ignore_index=True)
    test_pool  = pd.concat([ds_data[k] for k in test_keys],  ignore_index=True)

    missing = [c for c in fcols if c not in train_pool.columns]
    if missing:
        print(f"  [skip] columns {missing} not available")
        return

    classes = label_order

    if cross_ds:
        rng = np.random.default_rng(SEED)
        n_sample = max(1, int(len(train_pool) * frac))
        train_idx = rng.choice(len(train_pool), size=n_sample, replace=False)
        train_df  = train_pool.iloc[train_idx]
        test_df   = test_pool
    else:
        X = train_pool[fcols].values
        y = train_pool["label"].values
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1.0 - frac, random_state=SEED)
        tr_idx, te_idx = next(sss.split(X, y))
        train_df = train_pool.iloc[tr_idx]
        test_df  = train_pool.iloc[te_idx]

    clf = make_classifier()
    clf.fit(train_df[fcols].values, train_df["label"].values)
    y_pred = clf.predict(test_df[fcols].values)

    pred_df = test_df[["filename", "dataset", "label"]].copy()
    pred_df = pred_df.rename(columns={"label": "true_label"})
    pred_df["pred_label"] = y_pred

    # Save all predictions
    pred_csv = EVAL_DIR / f"predictions_{scenario_name}_{suffix}.csv"
    pred_df.to_csv(pred_csv, index=False)
    print(f"  saved {len(pred_df)} predictions → {pred_csv.name}")

    # Print confusion summary
    errors = pred_df[pred_df["true_label"] != pred_df["pred_label"]]
    print(f"  {len(errors)}/{len(pred_df)} misclassified ({100*len(errors)/len(pred_df):.1f}%)")
    if len(errors) > 0:
        conf = errors.groupby(["true_label", "pred_label"], observed=True).size().reset_index(name="n")
        conf = conf.sort_values("n", ascending=False)
        for _, row in conf.head(10).iterrows():
            t = LABEL_SHORT.get(row["true_label"], row["true_label"])
            p = LABEL_SHORT.get(row["pred_label"], row["pred_label"])
            print(f"    true={t:5s} → pred={p:5s}  n={row['n']}")

    _plot_errors(errors, scenario_name, suffix, label_order=label_order)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix",     default="A_zrecon")
    parser.add_argument("--frac",       type=float, default=0.75)
    parser.add_argument("--scenario",   default="all", help="Scenario name or 'all'")
    parser.add_argument("--variant",    default=None,
                        help="Override feature variant (zrecon/zproj/both). "
                             "Inferred from suffix if not set.")
    parser.add_argument("--five_class", action="store_true",
                        help="Include 'No adhesion' as a 5th class (appends _5cls to suffix)")
    args = parser.parse_args()

    label_order = FA_LABEL_ORDER_5 if args.five_class else FA_LABEL_ORDER_4
    suffix = args.suffix + ("_5cls" if args.five_class else "")

    # Infer variant from suffix if not given
    if args.variant:
        variant = args.variant
    elif "zproj" in args.suffix:
        variant = "zproj"
    else:
        variant = "zrecon"
    fcols = _feat_cols(variant)
    print(f"[config] suffix={suffix}  frac={args.frac}  variant={variant}  "
          f"fcols={len(fcols)}d  classes={len(label_order)}")

    scenarios_to_run = [
        s for s in SCENARIOS
        if args.scenario == "all" or s[0] == args.scenario
    ]

    for sc_name, train_ks, test_ks, cross in scenarios_to_run:
        try:
            run_scenario(sc_name, train_ks, test_ks, cross,
                         fcols, args.frac, suffix, label_order)
        except Exception as e:
            print(f"  [error] {sc_name}: {e}")


if __name__ == "__main__":
    main()
