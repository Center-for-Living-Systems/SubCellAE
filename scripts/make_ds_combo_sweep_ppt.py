#!/usr/bin/env python3
"""
make_ds_combo_sweep_ppt.py

PPT summarising the 15-combination dataset sweep for ConAE
(enlcrop / sc2 / nL1 / λ=0.25).

Slides
------
  1. Title
  2. Experiment overview (combos + training params)
  3. Summary table  — mean nL1 per model × dataset
  4. Violin plots   — 4 singles
  5. Violin plots   — pairs A  (vinc+nih3t3, vinc+ppax, vinc+pfak)
  6. Violin plots   — pairs B  (nih3t3+ppax, nih3t3+pfak, ppax+pfak)
  7. Violin plots   — 4 triples
  8. Violin plot    — all-4 datasets
  9. UMAP condition — 4 singles
 10. UMAP condition — pairs A
 11. UMAP condition — pairs B
 12. UMAP condition — 4 triples
 13. UMAP condition — all-4 datasets

Usage
-----
  python scripts/make_ds_combo_sweep_ppt.py [--out slides_ds_combo_sweep.pptx]

  If cross_dataset_recon_metrics.csv does not yet exist (eval job still
  running) the table slide is skipped automatically.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# ── paths ─────────────────────────────────────────────────────────────────────

RUNS = Path("/net/projects/CLS/lding/data/fa_data_analysis"
            "/ae_results/contrastive_run/ds_combo_enlcrop_sc2")

COMBO_LIST_TXT = Path("config/contrastive_config/ds_combo/combo_list.txt")

# ── combo groups ──────────────────────────────────────────────────────────────

SINGLES  = ["vinc", "nih3t3", "ppax", "pfak"]
PAIRS_A  = ["vinc_nih3t3", "vinc_ppax", "vinc_pfak"]
PAIRS_B  = ["nih3t3_ppax", "nih3t3_pfak", "ppax_pfak"]
TRIPLES  = ["vinc_nih3t3_ppax", "vinc_nih3t3_pfak",
            "vinc_ppax_pfak",   "nih3t3_ppax_pfak"]
ALL_DS   = ["vinc_nih3t3_ppax_pfak"]

GROUPS_VIOLIN = [SINGLES, PAIRS_A, PAIRS_B, TRIPLES, ALL_DS]
GROUPS_UMAP   = [SINGLES, PAIRS_A, PAIRS_B, TRIPLES, ALL_DS]

DATASETS_ORDERED = ["vinc", "pfak", "ppax", "nih3t3"]

DS_LABEL = {
    "vinc":   "ds1 vinc\n(train)",
    "pfak":   "ds2 pfak",
    "ppax":   "ds3 ppax",
    "nih3t3": "ds4 nih3t3",
}

REPEAT = {"vinc": 1, "nih3t3": 4, "ppax": 4, "pfak": 8}

# ── slide geometry ────────────────────────────────────────────────────────────

W_IN, H_IN = 13.33, 7.5
MARGIN = 0.25
HEADER_H = 0.68   # space consumed by title bar

# ── colour palette ────────────────────────────────────────────────────────────

C_TITLE  = RGBColor(0x1F, 0x2D, 0x3D)
C_ACCENT = RGBColor(0x2E, 0x86, 0xC1)
C_WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
C_GRAY   = RGBColor(0x66, 0x66, 0x66)
C_LGRAY  = RGBColor(0xAA, 0xAA, 0xAA)
C_LGRAY2 = RGBColor(0xEE, 0xEE, 0xEE)
C_ORANGE = RGBColor(0xE6, 0x7E, 0x22)
C_GREEN  = RGBColor(0x27, 0xAE, 0x60)

DS_COLOR = {
    "vinc":   RGBColor(0x4C, 0x72, 0xB0),
    "pfak":   RGBColor(0xC4, 0x4E, 0x52),
    "ppax":   RGBColor(0x55, 0xA8, 0x68),
    "nih3t3": RGBColor(0xDD, 0x84, 0x52),
}

# ── pptx helpers ──────────────────────────────────────────────────────────────

def _px(inches: float): return Inches(inches)


def _add_slide(prs: Presentation):
    return prs.slides.add_slide(prs.slide_layouts[6])


def _textbox(slide, text, left, top, width, height,
             font_size=11, bold=False, color=C_TITLE,
             align=PP_ALIGN.LEFT, italic=False, wrap=True):
    tb = slide.shapes.add_textbox(_px(left), _px(top), _px(width), _px(height))
    tf = tb.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return tb


def _slide_header(slide, title, subtitle=None, color=C_TITLE):
    """Colored top bar with title (and optional subtitle)."""
    bar = slide.shapes.add_shape(
        1, _px(0), _px(0), _px(W_IN), _px(0.62))
    bar.fill.solid()
    bar.fill.fore_color.rgb = color
    bar.line.fill.background()

    _textbox(slide, title, 0.25, 0.05, W_IN - 0.5, 0.45,
             font_size=22, bold=True, color=C_WHITE)
    if subtitle:
        _textbox(slide, subtitle, 0.25, 0.47, W_IN - 0.5, 0.22,
                 font_size=11, color=RGBColor(0xBB, 0xCC, 0xEE))


def _add_image_bytes(slide, img_bytes: bytes, left, top, width=None, height=None):
    buf = io.BytesIO(img_bytes)
    return slide.shapes.add_picture(
        buf, _px(left), _px(top),
        width=_px(width) if width is not None else None,
        height=_px(height) if height is not None else None,
    )


def _fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def _load_png(path: Path) -> np.ndarray | None:
    if path.exists():
        return np.array(Image.open(str(path)).convert("RGB"))
    return None


# ── composite figure builders ─────────────────────────────────────────────────

def _violin_grid(combos: list[str], n_cols: int | None = None) -> bytes:
    """Side-by-side violin plots for the given combos."""
    n = len(combos)
    if n_cols is None:
        n_cols = min(n, 4)
    n_rows = (n + n_cols - 1) // n_cols

    # Load violin PNGs
    imgs = []
    labels = []
    for c in combos:
        p = RUNS / c / "cross_dataset_recon_nl1.png"
        img = _load_png(p)
        imgs.append(img)
        labels.append(c.replace("_", "+"))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 5.5, n_rows * 4.5),
                             facecolor="white")
    axes = np.array(axes).reshape(n_rows, n_cols)

    for i, (img, lbl) in enumerate(zip(imgs, labels)):
        r, c = i // n_cols, i % n_cols
        ax = axes[r, c]
        if img is not None:
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "not yet generated",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="gray")
        ax.set_title(lbl, fontsize=11, fontweight="bold", pad=4)
        ax.axis("off")

    # hide unused axes
    for i in range(n, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].axis("off")

    fig.tight_layout(pad=0.8)
    return _fig_to_bytes(fig)


def _umap_grid(combos: list[str], n_cols: int | None = None) -> bytes:
    """Side-by-side UMAP condition plots."""
    n = len(combos)
    if n_cols is None:
        n_cols = min(n, 4)
    n_rows = (n + n_cols - 1) // n_cols

    imgs = []
    labels = []
    for c in combos:
        p = RUNS / c / "eval" / "umap_combo_condition.png"
        img = _load_png(p)
        imgs.append(img)
        labels.append(c.replace("_", "+"))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4.0, n_rows * 3.8),
                             facecolor="white")
    axes = np.array(axes).reshape(n_rows, n_cols)

    for i, (img, lbl) in enumerate(zip(imgs, labels)):
        r, c = i // n_cols, i % n_cols
        ax = axes[r, c]
        if img is not None:
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "not yet generated",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="gray")
        ax.set_title(lbl, fontsize=11, fontweight="bold", pad=4)
        ax.axis("off")

    for i in range(n, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].axis("off")

    fig.tight_layout(pad=0.8)
    return _fig_to_bytes(fig)


# ── slide builders ────────────────────────────────────────────────────────────

def _slide_title(prs: Presentation) -> None:
    slide = _add_slide(prs)
    bg = slide.background
    bg.fill.solid()
    bg.fill.fore_color.rgb = C_TITLE

    _textbox(slide, "Dataset Combination ConAE Sweep",
             1.0, 1.5, W_IN - 2.0, 1.2,
             font_size=38, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)

    _textbox(slide,
             "enlcrop · sc2 · nL1 · λ = 0.25  |  15 dataset combinations of "
             "{vinc, nih3t3, ppax, pfak}",
             1.0, 2.85, W_IN - 2.0, 0.6,
             font_size=18, color=RGBColor(0xAA, 0xCC, 0xFF),
             align=PP_ALIGN.CENTER)

    items = [
        "Single datasets: vinc (ds1), nih3t3 (ds2), ppax (ds3), pfak (ds4)",
        "All pairs (6), triples (4), and the full 4-dataset combination",
        "Balanced sampling: vinc×1, nih3t3×4, ppax×4, pfak×8 "
        "(target ≈ 27k patches from each dataset)",
        "Evaluated on all 4 datasets — violin plots of normalised L1 "
        "reconstruction error",
    ]
    y = 3.8
    for item in items:
        _textbox(slide, f"• {item}", 1.5, y, W_IN - 3.0, 0.42,
                 font_size=14, color=RGBColor(0xCC, 0xDD, 0xFF))
        y += 0.45

    _textbox(slide,
             "Latent dim 12  ·  Proj dim 8  ·  500 epochs  ·  Adam cosine LR",
             1.0, 6.5, W_IN - 2.0, 0.4,
             font_size=12, italic=True, color=RGBColor(0x88, 0xAA, 0xDD),
             align=PP_ALIGN.CENTER)


def _slide_overview(prs: Presentation) -> None:
    slide = _add_slide(prs)
    _slide_header(slide,
                  "Experiment Overview — 15 Dataset Combinations",
                  subtitle="Balanced oversampling · enlcrop/sc2 augmentation · "
                           "ConAE nL1 loss · λ_contrast = 0.25")

    # combo table (left side)
    combos = SINGLES + PAIRS_A + PAIRS_B + TRIPLES + ALL_DS
    groups = (["single"] * 4 + ["pair"] * 6 + ["triple"] * 4 + ["all"] * 1)
    col_x  = [MARGIN, 1.6, 3.5, 5.4, 7.3]
    col_w  = [1.3,    1.8, 1.8, 1.8, 1.8]
    headers = ["#", "Combo", "vinc", "ppax/pfak", "nih3t3"]
    row_h  = 0.38
    y0     = HEADER_H

    for ci, (hdr, cw, cx) in enumerate(zip(headers, col_w, col_x)):
        box = slide.shapes.add_shape(1, _px(cx), _px(y0), _px(cw), _px(row_h))
        box.fill.solid(); box.fill.fore_color.rgb = C_TITLE
        box.line.fill.background()
        _textbox(slide, hdr, cx + 0.04, y0 + 0.06, cw - 0.08, row_h - 0.08,
                 font_size=9, bold=True, color=C_WHITE)

    for i, (combo, grp) in enumerate(zip(combos, groups)):
        y = y0 + row_h * (i + 1)
        bg = C_LGRAY2 if i % 2 == 0 else C_WHITE
        parts = set(combo.split("_"))
        row_vals = [
            str(i + 1),
            combo.replace("_", "+"),
            f"×{REPEAT['vinc']}" if "vinc" in parts else "—",
            "+".join(f"×{REPEAT[d]}" for d in ["ppax", "pfak"] if d in parts) or "—",
            f"×{REPEAT['nih3t3']}" if "nih3t3" in parts else "—",
        ]
        for ci, (val, cw, cx) in enumerate(zip(row_vals, col_w, col_x)):
            box = slide.shapes.add_shape(1, _px(cx), _px(y), _px(cw), _px(row_h))
            box.fill.solid(); box.fill.fore_color.rgb = bg
            box.line.fill.background()
            color = C_ACCENT if ci == 1 else C_GRAY
            _textbox(slide, val, cx + 0.04, y + 0.05, cw - 0.08, row_h - 0.08,
                     font_size=8, color=color)

    # training param box (right side)
    rx = 9.3
    _textbox(slide, "Training Settings", rx, HEADER_H, 3.8, 0.38,
             font_size=13, bold=True, color=C_TITLE)
    params = [
        ("Architecture",   "ConAE — contrastive autoencoder"),
        ("Loss",           "nL1 recon + SC2 contrastive (λ=0.25)"),
        ("Augmentation",   "EnlargedJitterCrop (enlcrop, ±15°, ±4px)\n"
                           "+ sc2 intensity jitter"),
        ("Latent / Proj",  "12 / 8"),
        ("Epochs",         "500"),
        ("Optimizer",      "Adam, cosine LR decay"),
        ("Patch size",     "32×32 px  (context 58×58)"),
        ("Balanced",       "Repeat minority datasets so each\n"
                           "contributes ~27k patches"),
    ]
    y = HEADER_H + 0.42
    for label, val in params:
        _textbox(slide, label + ":", rx, y, 1.6, 0.38,
                 font_size=9, bold=True, color=C_TITLE)
        _textbox(slide, val, rx + 1.65, y, 2.15, 0.42,
                 font_size=9, color=C_GRAY)
        y += 0.42

    _textbox(slide,
             "Evaluation: all 4 datasets (vinc=train, pfak/ppax/nih3t3=test)\n"
             "Metric: normalised L1 = L1 / mean|raw|  (lower = better)",
             rx, y + 0.1, 3.8, 0.65,
             font_size=9, italic=True, color=C_ACCENT)


def _slide_mean_table(prs: Presentation) -> None:
    csv_path = RUNS / "cross_dataset_recon_metrics.csv"
    if not csv_path.exists():
        print(f"  [SKIP] table slide — {csv_path} not found yet")
        return

    df = pd.read_csv(csv_path)
    if "variant" not in df.columns or "recon_nl1" not in df.columns:
        print("  [SKIP] table slide — unexpected CSV columns")
        return

    combos = SINGLES + PAIRS_A + PAIRS_B + TRIPLES + ALL_DS

    # mean nL1 per (variant, dataset)
    mean_df = (df.groupby(["variant", "dataset"])["recon_nl1"]
               .mean()
               .reset_index())

    # Build matrix: rows=combos, cols=datasets
    table = {}
    for _, row in mean_df.iterrows():
        table.setdefault(row["variant"], {})[row["dataset"]] = row["recon_nl1"]

    slide = _add_slide(prs)
    _slide_header(slide,
                  "Summary — Mean Normalised L1 per Model × Dataset",
                  subtitle="Lower = better  ·  vinc = train set  ·  "
                           "pfak / ppax / nih3t3 = unseen test sets",
                  color=C_TITLE)

    datasets = DATASETS_ORDERED
    n_cols = 1 + len(datasets)   # combo name + 4 dataset cols
    col_labels = ["Combo"] + [DS_LABEL[d] for d in datasets]
    col_w = [3.2] + [2.1] * len(datasets)
    col_x: list[float] = []
    x = MARGIN
    for cw in col_w:
        col_x.append(x)
        x += cw

    row_h  = 0.35
    y0     = HEADER_H

    # header
    for ci, (hdr, cw, cx) in enumerate(zip(col_labels, col_w, col_x)):
        box = slide.shapes.add_shape(1, _px(cx), _px(y0), _px(cw), _px(row_h))
        box.fill.solid()
        box.fill.fore_color.rgb = C_TITLE if ci == 0 else DS_COLOR.get(
            datasets[ci - 1] if ci > 0 else "vinc", C_TITLE)
        box.line.fill.background()
        _textbox(slide, hdr, cx + 0.04, y0 + 0.04, cw - 0.08, row_h - 0.08,
                 font_size=9, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)

    # collect all values for heatmap coloring
    all_vals = [v for row in [table.get(c, {}) for c in combos]
                for v in row.values() if v is not None]
    vmin = min(all_vals) if all_vals else 0.0
    vmax = max(all_vals) if all_vals else 1.0

    def _heat_color(val: float) -> RGBColor:
        t = (val - vmin) / max(vmax - vmin, 1e-6)
        r = int(255 * (0.2 + 0.6 * t))
        g = int(255 * (0.9 - 0.5 * t))
        b = int(255 * (0.9 - 0.7 * t))
        return RGBColor(
            max(0, min(255, r)),
            max(0, min(255, g)),
            max(0, min(255, b)),
        )

    for i, combo in enumerate(combos):
        y = y0 + row_h * (i + 1)
        row_data = table.get(combo, {})
        bg = C_LGRAY2 if i % 2 == 0 else C_WHITE

        # combo name cell
        box = slide.shapes.add_shape(1, _px(col_x[0]), _px(y),
                                     _px(col_w[0]), _px(row_h))
        box.fill.solid(); box.fill.fore_color.rgb = bg
        box.line.fill.background()
        _textbox(slide, combo.replace("_", "+"),
                 col_x[0] + 0.04, y + 0.04, col_w[0] - 0.08, row_h - 0.06,
                 font_size=9, bold=False, color=C_TITLE)

        # metric cells
        for ci, ds in enumerate(datasets, start=1):
            val = row_data.get(ds)
            cx  = col_x[ci]
            cw  = col_w[ci]
            box = slide.shapes.add_shape(1, _px(cx), _px(y), _px(cw), _px(row_h))
            if val is not None:
                box.fill.solid()
                box.fill.fore_color.rgb = _heat_color(val)
            else:
                box.fill.solid(); box.fill.fore_color.rgb = bg
            box.line.fill.background()
            txt = f"{val:.3f}" if val is not None else "—"
            _textbox(slide, txt, cx + 0.04, y + 0.04, cw - 0.08, row_h - 0.06,
                     font_size=9, color=C_TITLE, align=PP_ALIGN.CENTER)

    y_note = y0 + row_h * (len(combos) + 1) + 0.1
    _textbox(slide,
             "Color scale: green = low nL1 (better reconstruction); "
             "red = high nL1 (worse); scaled per table.",
             MARGIN, y_note, W_IN - 2 * MARGIN, 0.35,
             font_size=9, italic=True, color=C_GRAY)


def _slide_violin_group(prs: Presentation, combos: list[str],
                        group_label: str) -> None:
    slide = _add_slide(prs)
    n = len(combos)
    n_cols = min(n, 4)
    _slide_header(slide,
                  f"Violin Plots — {group_label}",
                  subtitle="Normalised L1 per group  "
                            "(vinc FA types | pfak / ppax / nih3t3 conditions)",
                  color=C_ACCENT)

    img = _violin_grid(combos, n_cols=n_cols)
    avail_h = H_IN - HEADER_H - 0.1
    _add_image_bytes(slide, img, MARGIN, HEADER_H + 0.05,
                     width=W_IN - 2 * MARGIN)


def _slide_umap_group(prs: Presentation, combos: list[str],
                      group_label: str) -> None:
    slide = _add_slide(prs)
    n = len(combos)
    n_cols = min(n, 4)
    _slide_header(slide,
                  f"UMAP Condition — {group_label}",
                  subtitle="z_proj latent space coloured by dataset+condition  "
                            "(UMAP fitted on eval set)",
                  color=C_GREEN)

    img = _umap_grid(combos, n_cols=n_cols)
    _add_image_bytes(slide, img, MARGIN, HEADER_H + 0.05,
                     width=W_IN - 2 * MARGIN)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="slides_ds_combo_sweep.pptx")
    args = parser.parse_args()

    prs = Presentation()
    prs.slide_width  = Inches(W_IN)
    prs.slide_height = Inches(H_IN)

    group_labels_violin = [
        "Singles  (vinc · nih3t3 · ppax · pfak)",
        "Pairs A  (vinc+nih3t3 · vinc+ppax · vinc+pfak)",
        "Pairs B  (nih3t3+ppax · nih3t3+pfak · ppax+pfak)",
        "Triples  (all 4 triples)",
        "All 4 datasets",
    ]
    group_labels_umap = [
        "Singles  (vinc · nih3t3 · ppax · pfak)",
        "Pairs A  (vinc+nih3t3 · vinc+ppax · vinc+pfak)",
        "Pairs B  (nih3t3+ppax · nih3t3+pfak · ppax+pfak)",
        "Triples  (all 4 triples)",
        "All 4 datasets",
    ]

    steps = [
        ("Title",              _slide_title,      (prs,)),
        ("Overview",           _slide_overview,   (prs,)),
        ("Mean nL1 table",     _slide_mean_table, (prs,)),
    ]
    for combos, lbl in zip(GROUPS_VIOLIN, group_labels_violin):
        steps.append((f"Violin {lbl}", _slide_violin_group,
                      (prs, combos, lbl)))
    for combos, lbl in zip(GROUPS_UMAP, group_labels_umap):
        steps.append((f"UMAP {lbl}", _slide_umap_group,
                      (prs, combos, lbl)))

    for label, fn, args_ in steps:
        print(f"  {label} …", flush=True)
        fn(*args_)

    out = Path(args.out)
    prs.save(str(out))
    print(f"\nSaved → {out}  ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
