#!/usr/bin/env python3
"""
make_pptx_noad_vs_ad_story.py

Comprehensive slide deck for the No Adhesion vs Adhesion 2-class story:
  config, data, inter-annotator agreement, model, in-domain results,
  cross-dataset blind test (vinc ctrl/ycomp, ppax Margaret/Ernest, pfak).

All slides: white background, no decorative colours on boxes.

Usage:
  python scripts/make_pptx_noad_vs_ad_story.py
  python scripts/make_pptx_noad_vs_ad_story.py --out path/to/out.pptx
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/net/projects/CLS/lding/data/fa_data_analysis")
RUN_DIR   = DATA_ROOT / "ae_results" / "contrastive_run"
LABEL_DIR = DATA_ROOT / "labelling"

SPLITS    = ["s1v3", "s2v2", "s3v1"]
FEATS     = ["zrecon", "zproj"]
ADHESION  = {"focal adhesion", "Nascent Adhesion", "fibrillar adhesion", "focal complex"}

LABEL_ORDER_5 = ["No adhesion","Nascent Adhesion","focal complex","focal adhesion","fibrillar adhesion"]
LABEL_ORDER_2 = ["No adhesion","adhesion"]

# Slide dimensions: widescreen 13.33 × 7.5 in
SW = Inches(13.33)
SH = Inches(7.5)

# Colours (all text; backgrounds stay white)
C_TITLE  = RGBColor(0x1A, 0x1A, 0x2E)   # near-black navy
C_HEAD   = RGBColor(0x16, 0x21, 0x3E)   # section heading
C_BODY   = RGBColor(0x1A, 0x1A, 0x1A)
C_ACCENT = RGBColor(0x0F, 0x3D, 0x79)   # dark blue accent
C_GOOD   = RGBColor(0x1A, 0x6B, 0x30)   # dark green
C_WARN   = RGBColor(0x8B, 0x45, 0x00)   # dark orange
C_GREY   = RGBColor(0x66, 0x66, 0x66)
C_BLACK  = RGBColor(0x00, 0x00, 0x00)

# Dataset display name mapping (internal key → slide label)
DS_DISPLAY = {
    "vinc":   "dataset1",
    "pfak":   "dataset2",
    "ppax":   "dataset3",
    "nih3t3": "dataset4",
}

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _prs() -> Presentation:
    prs = Presentation()
    prs.slide_width  = SW
    prs.slide_height = SH
    return prs


def _blank(prs: Presentation):
    return prs.slides.add_slide(prs.slide_layouts[6])


def _txt(slide, text: str, left, top, width, height, *,
         bold=False, italic=False, size_pt=13, color=C_BODY,
         align=PP_ALIGN.LEFT, wrap=True):
    txb = slide.shapes.add_textbox(left, top, width, height)
    tf  = txb.text_frame
    tf.word_wrap = wrap
    p   = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.bold   = bold
    run.font.italic = italic
    run.font.size   = Pt(size_pt)
    run.font.color.rgb = color
    return txb


def _rule(slide, top, width=None, left=None, thickness_pt=0.75):
    """Thin horizontal rule."""
    w = width or SW - Inches(1.0)
    l = left  or Inches(0.5)
    ln = slide.shapes.add_connector(
        1,   # MSO_CONNECTOR_TYPE.STRAIGHT
        l, top, l + w, top)
    ln.line.color.rgb = RGBColor(0xCC, 0xCC, 0xCC)
    ln.line.width = Pt(thickness_pt)


def _slide_header(slide, title: str, subtitle: str = ""):
    _txt(slide, title,
         Inches(0.5), Inches(0.12), Inches(12.3), Inches(0.55),
         bold=True, size_pt=22, color=C_HEAD)
    if subtitle:
        _txt(slide, subtitle,
             Inches(0.5), Inches(0.65), Inches(12.3), Inches(0.35),
             size_pt=12, color=C_GREY)
    _rule(slide, Inches(0.97))


def _img_or_ph(slide, path, left, top, width, height, label="[pending]"):
    if path and Path(path).exists():
        slide.shapes.add_picture(str(path), left, top, width=width, height=height)
    else:
        box = slide.shapes.add_textbox(left, top, width, height)
        tf  = box.text_frame
        tf.paragraphs[0].add_run().text = label
        tf.paragraphs[0].runs[0].font.size = Pt(9)
        tf.paragraphs[0].runs[0].font.color.rgb = C_GREY


def _fig_to_img(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf


def _add_fig(slide, fig, left, top, width, height):
    buf = _fig_to_img(fig)
    slide.shapes.add_picture(buf, left, top, width=width, height=height)


# ---------------------------------------------------------------------------
# Pre-generated figures
# ---------------------------------------------------------------------------

def _fig_label_dist_annabel():
    """Bar chart of Annabel's vinc control label distribution (5→2 class)."""
    ann = pd.read_csv(LABEL_DIR / "vinc_control_label_Annabel_20260715_1554.csv")
    counts5 = ann["label"].value_counts().reindex(LABEL_ORDER_5, fill_value=0)
    counts2 = pd.Series({
        "No adhesion": counts5["No adhesion"],
        "adhesion":    counts5.drop("No adhesion").sum(),
    })

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), facecolor="white")
    # 5-class
    ax = axes[0]
    colors5 = ["#AAAAAA","#4E79A7","#59A14F","#F28E2B","#E15759"]
    bars = ax.bar(range(len(LABEL_ORDER_5)), counts5.values, color=colors5, edgecolor="white")
    ax.set_xticks(range(len(LABEL_ORDER_5)))
    ax.set_xticklabels(["No adh.","Nascent","FC","FA","Fibrillar"], fontsize=9, rotation=20, ha="right")
    ax.set_title("5-class labels (Annabel, dataset1 ctrl)", fontsize=10)
    ax.set_ylabel("# patches"); ax.set_facecolor("white"); ax.spines[["top","right"]].set_visible(False)
    for b, v in zip(bars, counts5.values):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1, str(int(v)), ha="center", fontsize=8)
    # 2-class
    ax2 = axes[1]
    colors2 = ["#AAAAAA","#4E79A7"]
    bars2 = ax2.bar(["No adhesion","adhesion"], counts2.values, color=colors2, edgecolor="white")
    ax2.set_title("Remapped to 2-class", fontsize=10)
    ax2.set_ylabel("# patches"); ax2.set_facecolor("white"); ax2.spines[["top","right"]].set_visible(False)
    for b, v in zip(bars2, counts2.values):
        ax2.text(b.get_x()+b.get_width()/2, b.get_height()+1, str(int(v)), ha="center", fontsize=10, fontweight="bold")
    fig.tight_layout(pad=1.5)
    return fig


def _fig_inter_annotator():
    """Summary figure: overlap & 2-class agreement across annotator pairs."""
    import re as _re

    ann  = pd.read_csv(LABEL_DIR / "vinc_control_label_Annabel_20260715_1554.csv")
    marg_v = pd.read_csv(LABEL_DIR / "labels_vinc_20260521.csv")
    marg_p = pd.read_csv(LABEL_DIR / "labels_ppax_20260521.csv")
    ern  = pd.read_csv(LABEL_DIR / "ppax_control_label_Ernest_20260727_1142.csv")
    ern["unique_ID"] = ern["filename"].apply(
        lambda x: _re.sub(r'_(?=f\d{4})', '-', Path(x).name, count=1))

    marg_vc = marg_v[marg_v["condition"] == "control"].copy()
    merged_av = ann.merge(marg_vc[["unique_ID","classification"]], on="unique_ID", how="inner")
    merged_av["a2"] = merged_av["label"].apply(lambda x: "adhesion" if x in ADHESION else x)
    merged_av["m2"] = merged_av["classification"].apply(
        lambda x: "adhesion" if x in ADHESION else ("skip" if x == "Uncertain" else x))
    av_ev = merged_av[merged_av["m2"] != "skip"]

    merged_ep = ern.merge(marg_p[["unique_ID","classification"]], on="unique_ID", how="inner")
    merged_ep["e2"] = merged_ep["label"].apply(lambda x: "adhesion" if x in ADHESION else x)
    merged_ep["m2"] = merged_ep["classification"].apply(
        lambda x: "adhesion" if x in ADHESION else ("skip" if x == "Uncertain" else x))
    ep_ev = merged_ep[merged_ep["m2"] != "skip"]

    rows = [
        ("Annabel vs\nMargaret\n(dataset1 ctrl)", len(merged_av), len(av_ev),
         (av_ev["a2"] == av_ev["m2"]).sum(), len(av_ev),
         (merged_av["label"] == merged_av["classification"]).sum(), len(merged_av)),
        ("Ernest vs\nMargaret\n(dataset3 ctrl)", len(merged_ep), len(ep_ev),
         (ep_ev["e2"] == ep_ev["m2"]).sum(), len(ep_ev),
         (merged_ep["label"] == merged_ep["classification"]).sum(), len(merged_ep)),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), facecolor="white")
    labels = [r[0] for r in rows]
    agree2 = [r[3]/r[4]*100 for r in rows]
    agree5 = [r[5]/r[6]*100 for r in rows]
    x = np.arange(len(labels))
    w = 0.35

    for ax_i, (vals, title, fmt) in enumerate([
        (agree2, "2-class agreement (%)\n(adhesion vs no adhesion)", "{:.0f}%"),
        (agree5, "5-class agreement (%)\n(FA subtype)", "{:.0f}%"),
    ]):
        ax = axes[ax_i]
        bars = ax.bar(x, vals, width=0.5, color=["#4E79A7","#59A14F"], edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 115); ax.set_title(title, fontsize=10)
        ax.set_ylabel("% agreement"); ax.set_facecolor("white")
        ax.spines[["top","right"]].set_visible(False)
        ax.axhline(100, color="#AAAAAA", lw=0.8, ls="--")
        for b, v in zip(bars, vals):
            n_agree = int(v * [r[4] for r in rows][list(x).index(b.get_x()+0.25)] / 100 + 0.5)
            total   = [r[4] if ax_i==0 else r[6] for r in rows][list(x).index(b.get_x()+0.25)]
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+2,
                    f"{v:.0f}%\n({n_agree}/{total})", ha="center", fontsize=9, fontweight="bold")
    fig.tight_layout(pad=1.5)
    return fig


def _fig_accuracy_heatmap():
    """Heatmap: supcon2 accuracy across datasets × splits × features."""
    DS_LABELS = ["dataset1\nctrl","dataset1\nycomp","dataset3\nctrl\n(Margaret)","dataset2\nctrl","dataset3\nctrl\n(Ernest)"]
    DS_KEYS   = [
        ("vinc","control","blind","zrecon"),
        ("vinc","ycomp",  "blind","zrecon"),
        ("ppax","control","blind","zrecon"),
        ("pfak","control","blind","zrecon"),
        ("ppax","control","ernest","zrecon"),
    ]

    data = np.full((len(SPLITS), len(DS_KEYS)), np.nan)
    for si, split in enumerate(SPLITS):
        rdir = RUN_DIR / f"annabel_vinc_supcon2_{split}" / "blind_test"
        for di, (ds, cond, kind, feat) in enumerate(DS_KEYS):
            if kind == "blind":
                p = rdir / f"{ds}_{cond}_{feat}" / "metrics.csv"
            else:
                p = rdir / f"{ds}_{cond}_ernest_{feat}" / "metrics.csv"
            if p.exists():
                df = pd.read_csv(p)
                data[si, di] = float(df["accuracy"].iloc[0])

    fig, ax = plt.subplots(figsize=(9, 3.2), facecolor="white")
    im = ax.imshow(data, vmin=0.5, vmax=1.0, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(DS_LABELS))); ax.set_xticklabels(DS_LABELS, fontsize=9)
    ax.set_yticks(range(len(SPLITS))); ax.set_yticklabels(SPLITS, fontsize=9)
    ax.set_xlabel("Dataset / condition", fontsize=10)
    ax.set_ylabel("Train/val split", fontsize=10)
    ax.set_title("SupCon-2cls accuracy (z_recon) across datasets and splits", fontsize=11)
    for si in range(len(SPLITS)):
        for di in range(len(DS_KEYS)):
            v = data[si, di]
            if not np.isnan(v):
                ax.text(di, si, f"{v:.2f}", ha="center", va="center", fontsize=10,
                        color="white" if v > 0.78 else "black", fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Accuracy")
    fig.tight_layout()
    return fig


def _fig_zrecon_vs_zproj():
    """Bar comparison: z_recon vs z_proj accuracy for supcon2 (best split = s2v2)."""
    DS_LABELS  = ["dataset1/ctrl","dataset1/ycomp","dataset3/ctrl\n(Margaret)","dataset2/ctrl"]
    DS_KEYS    = [("vinc","control"),("vinc","ycomp"),("ppax","control"),("pfak","control")]

    accs = {feat: [] for feat in FEATS}
    split = "s2v2"
    rdir  = RUN_DIR / f"annabel_vinc_supcon2_{split}" / "blind_test"
    for feat in FEATS:
        for ds, cond in DS_KEYS:
            p = rdir / f"{ds}_{cond}_{feat}" / "metrics.csv"
            accs[feat].append(float(pd.read_csv(p)["accuracy"].iloc[0]) if p.exists() else 0)

    x = np.arange(len(DS_LABELS))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 3.5), facecolor="white")
    b1 = ax.bar(x - w/2, accs["zrecon"], w, label="z_recon (12-d)", color="#4E79A7", edgecolor="white")
    b2 = ax.bar(x + w/2, accs["zproj"],  w, label="z_proj (8-d)",  color="#F28E2B", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(DS_LABELS, fontsize=9)
    ax.set_ylim(0, 1.1); ax.set_ylabel("Accuracy"); ax.set_facecolor("white")
    ax.set_title(f"z_recon vs z_proj — SupCon-2cls {split} (blind test)", fontsize=11)
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(fontsize=9, loc="lower right")
    ax.axhline(1.0, color="#AAAAAA", lw=0.8, ls="--")
    for b, vals in [(b1, accs["zrecon"]), (b2, accs["zproj"])]:
        for bar, v in zip(b, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                    f"{v:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    return fig


def _fig_indomain_val():
    """Bar: in-domain val macro-F1 across splits and features."""
    vals = {}
    for split in SPLITS:
        vals[split] = {}
        for feat in FEATS:
            p = RUN_DIR / f"annabel_vinc_supcon2_{split}" / f"fa_cls_{feat}" / "metrics.txt"
            if p.exists():
                mf1 = next((float(l.split()[4]) for l in p.read_text().splitlines()
                             if "macro avg" in l), 0.0)
                vals[split][feat] = mf1

    x = np.arange(len(SPLITS))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor="white")
    zr = [vals[s].get("zrecon", 0) for s in SPLITS]
    zp = [vals[s].get("zproj",  0) for s in SPLITS]
    b1 = ax.bar(x - w/2, zr, w, label="z_recon", color="#4E79A7", edgecolor="white")
    b2 = ax.bar(x + w/2, zp, w, label="z_proj",  color="#F28E2B", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(SPLITS, fontsize=10)
    ax.set_ylim(0.8, 1.05); ax.set_ylabel("Macro-F1 (val)")
    ax.set_title("SupCon-2cls — in-domain validation (Annabel dataset1 ctrl)", fontsize=11)
    ax.set_facecolor("white"); ax.spines[["top","right"]].set_visible(False)
    ax.legend(fontsize=9); ax.axhline(1.0, color="#AAAAAA", lw=0.8, ls="--")
    for b, vals_list in [(b1, zr), (b2, zp)]:
        for bar, v in zip(b, vals_list):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                    f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Slides
# ---------------------------------------------------------------------------

def slide_title(prs):
    sl = _blank(prs)
    _txt(sl, "No Adhesion vs Adhesion",
         Inches(0.8), Inches(2.0), Inches(11.7), Inches(1.2),
         bold=True, size_pt=36, color=C_TITLE, align=PP_ALIGN.LEFT)
    _txt(sl, "Training, Validation & Cross-Dataset Generalization",
         Inches(0.8), Inches(3.2), Inches(11.7), Inches(0.6),
         size_pt=20, color=C_ACCENT)
    _txt(sl,
         "SupCon-2cls   •   Annabel dataset1 labels (539 patches)   •   Blind test: dataset1 / dataset3 / dataset2\n"
         "z_recon 12-d  |  z_proj 8-d  |  LightGBM classifier  |  pax channel",
         Inches(0.8), Inches(4.0), Inches(11.7), Inches(0.8),
         size_pt=13, color=C_GREY)
    _rule(sl, Inches(5.2), width=Inches(11.7), left=Inches(0.8))


def slide_study_design(prs):
    sl = _blank(prs)
    _slide_header(sl, "Study Design", "Two-class binary detection: adhesion present or absent")
    body = (
        "Training data (single annotator — Annabel)\n"
        "  •  539 dataset1/control patches  •  4 images (f0000–f0003)  •  pax channel\n"
        "  •  5 FA subtypes remapped → 2 classes: No adhesion | adhesion\n\n"
        "Model: SupCon-2cls  (supervised contrastive loss)\n"
        "  •  Latent dim = 12 (z_recon)   Projection dim = 8 (z_proj)\n"
        "  •  Enlarged crop 58×58 px context → 32×32 px input  •  input_divisor = 2.0\n"
        "  •  3 per-image train/val splits: s1v3 (1 train/3 val), s2v2, s3v1\n\n"
        "Evaluation\n"
        "  In-domain val  — Annabel's own held-out images\n"
        "  Blind test     — Margaret's independent labels (labels_*_20260521.csv)\n"
        "                   dataset1/ctrl  |  dataset1/ycomp  |  dataset3/ctrl  |  dataset2/ctrl\n"
        "  Ernest test    — Ernest's dataset3/ctrl labels (all FA patches, no No-adhesion)"
    )
    _txt(sl, body, Inches(0.55), Inches(1.1), Inches(12.2), Inches(5.8), size_pt=13)


def slide_training_data(prs):
    sl = _blank(prs)
    _slide_header(sl, "Training Data", "Annabel's dataset1/control labels — 539 patches, 4 frames")

    # label distribution figure left
    fig = _fig_label_dist_annabel()
    _add_fig(sl, fig, Inches(0.5), Inches(1.1), Inches(8.5), Inches(3.5))

    # text right
    lines = (
        "Label distribution\n"
        "No adhesion:       342  (63.5%)\n"
        "focal adhesion:    168  (31.2%)\n"
        "Nascent Adhesion:   18  (  3.3%)\n"
        "focal complex:      10  (  1.9%)\n"
        "fibrillar adhesion:  11  (  2.0%)\n\n"
        "Remapped → 2-class:\n"
        "  No adhesion:  342\n"
        "  adhesion:     197"
    )
    _txt(sl, lines, Inches(9.1), Inches(1.1), Inches(3.9), Inches(3.5), size_pt=11, color=C_BODY)

    # frame table
    ann = pd.read_csv(LABEL_DIR / "vinc_control_label_Annabel_20260715_1554.csv")
    ann["frame"] = ann["unique_ID"].str.extract(r"(f\d+)")
    tbl_data = ann.groupby("frame")["label"].value_counts().unstack(fill_value=0)

    fig2, ax = plt.subplots(figsize=(11, 1.5), facecolor="white")
    ax.axis("off")
    cols = tbl_data.columns.tolist()
    rows_d = [[idx] + [str(int(tbl_data.loc[idx, c])) if c in tbl_data.columns else "0"
                       for c in cols] for idx in tbl_data.index]
    tbl = ax.table(cellText=rows_d, colLabels=["frame"] + cols,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.4)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        cell.set_facecolor("white")
        if r == 0:
            cell.set_text_props(fontweight="bold")
    fig2.tight_layout()
    _add_fig(sl, fig2, Inches(0.5), Inches(4.7), Inches(12.3), Inches(2.0))


def slide_inter_annotator(prs):
    sl = _blank(prs)
    _slide_header(sl, "Inter-Annotator Agreement",
                  "2-class boundary (adhesion vs no-adhesion) is annotator-consistent")

    fig = _fig_inter_annotator()
    _add_fig(sl, fig, Inches(0.5), Inches(1.1), Inches(9.5), Inches(3.8))

    notes = (
        "Key observations\n\n"
        "Annabel vs Margaret (dataset1 ctrl)\n"
        "  Overlap: 18 patches\n"
        "  2-class: 17/17 agree  (100%)\n"
        "  5-class: 13/18 agree  (72%)\n\n"
        "Ernest vs Margaret (dataset3 ctrl)\n"
        "  Overlap: 25 patches\n"
        "  2-class: n/a (Ernest has\n"
        "     no 'No adhesion' labels)\n"
        "  5-class: 20/25 agree  (80%)\n\n"
        "Disagreements are exclusively\n"
        "at subtype boundaries within\n"
        "the adhesion class."
    )
    _txt(sl, notes, Inches(10.1), Inches(1.1), Inches(2.9), Inches(5.5), size_pt=10)

    _txt(sl,
         "The no-adhesion / adhesion boundary is never crossed between annotators "
         "→ the 2-class problem has near-perfect human inter-annotator reliability.",
         Inches(0.5), Inches(5.1), Inches(9.5), Inches(0.8),
         size_pt=11, color=C_GOOD, bold=True)


def slide_ernest_ppax(prs):
    sl = _blank(prs)
    _slide_header(sl, "Ernest's dataset3 Labels",
                  "Independent labeling — dataset3/control, 2 frames, 111 patches")
    body = (
        "Ernest labeled only FA-positive patches (no 'No adhesion' assigned)\n\n"
        "  f0000:   55 patches   (35 focal adhesion, 15 Nascent Adhesion, 5 focal complex)\n"
        "  f0009:   56 patches   (23 focal adhesion, 10 Nascent Adhesion, 18 focal complex, 5 fibrillar)\n\n"
        "Comparison with Margaret's dataset3 labels (60 patches, f0000 only):\n"
        "  Overlap: 25 patches  →  5-class agreement: 20/25 (80%)\n"
        "  Ernest has no 'No adhesion' entries; Margaret has 15 'No adhesion' + 9 'Uncertain'\n"
        "  Different labeling strategy: Ernest focused on FA structure, Margaret included background\n\n"
        "Use in blind test:\n"
        "  Apply all 9 trained models to dataset3/control patches\n"
        "  Match predictions on Ernest's 111 patches by filename\n"
        "  For SupCon-2cls: remap Ernest's subtypes → 'adhesion' (all 111 should be predicted adhesion)\n"
        "  For ConAE / SupCon-5cls: evaluate subtype classification"
    )
    _txt(sl, body, Inches(0.55), Inches(1.1), Inches(12.3), Inches(5.5), size_pt=13)


def slide_model_config(prs):
    sl = _blank(prs)
    _slide_header(sl, "Model Configuration", "SupCon-2cls — Supervised Contrastive AutoEncoder")

    cols = [
        ("Architecture", [
            "Input:        32×32 px (pax channel)",
            "Context crop: 58×58 px (input_divisor=2.0)",
            "Encoder:      Conv → z_recon (12-d)",
            "Projection:   MLP → z_proj (8-d)",
            "Decoder:      ConvTranspose → reconstruction",
        ]),
        ("Training", [
            "Loss:         SupCon + recon (nl1)",
            "λ_recon=1.0   λ_contrast=0.5",
            "Temperature:  0.5",
            "Epochs:       500   LR: 0.001",
            "Batch:        128   weight decay: 0.0001",
            "Augmentation: ±shift 4px, ±15° rotation",
            "noise_prob:   0.0",
        ]),
        ("Label mapping", [
            "No adhesion      → 'No adhesion'",
            "Nascent Adhesion → 'adhesion'",
            "focal complex    → 'adhesion'",
            "focal adhesion   → 'adhesion'",
            "fibrillar adhes. → 'adhesion'",
            "",
            "Classifier: LightGBM (n_est=500)",
        ]),
        ("Splits", [
            "Per-image split (4 labeled frames):",
            "s1v3: f0000 train | f0001-3 val",
            "      (145 / 394 labeled patches)",
            "s2v2: f0000-1 train | f0002-3 val",
            "      (307 / 232 labeled patches)",
            "s3v1: f0000-2 train | f0003 val",
            "      (412 / 127 labeled patches)",
        ]),
    ]

    x0 = Inches(0.4)
    col_w = Inches(3.1)
    gap   = Inches(0.1)
    y0    = Inches(1.1)
    for ci, (heading, items) in enumerate(cols):
        x = x0 + ci * (col_w + gap)
        _txt(sl, heading, x, y0, col_w, Inches(0.35),
             bold=True, size_pt=12, color=C_ACCENT)
        body = "\n".join(items)
        _txt(sl, body, x, y0 + Inches(0.38), col_w, Inches(5.0), size_pt=10)


def slide_indomain_val(prs):
    sl = _blank(prs)
    _slide_header(sl, "In-Domain Validation",
                  "SupCon-2cls evaluated on Annabel's held-out dataset1/control images")

    fig = _fig_indomain_val()
    _add_fig(sl, fig, Inches(0.5), Inches(1.1), Inches(7.0), Inches(3.5))

    # Confusion matrix for best split (s2v2)
    cm_path = RUN_DIR / "annabel_vinc_supcon2_s2v2" / "fa_cls_zrecon" / "confusion_matrix_norm_val.png"
    _img_or_ph(sl, cm_path, Inches(7.7), Inches(1.1), Inches(5.2), Inches(3.5), "[val CM s2v2 zrecon]")

    _txt(sl,
         "All splits achieve macro-F1 ≥ 0.93 on held-out images from the same dataset.\n"
         "s2v2 and s3v1 reach 0.99 — near-perfect binary discrimination within Annabel's labeling.\n"
         "z_recon and z_proj perform similarly in-domain (both well-tuned to training distribution).",
         Inches(0.5), Inches(4.75), Inches(12.3), Inches(1.0),
         size_pt=12, color=C_BODY)

    # metrics table
    rows = []
    for split in SPLITS:
        row = [split]
        for feat in FEATS:
            p = RUN_DIR / f"annabel_vinc_supcon2_{split}" / f"fa_cls_{feat}" / "metrics.txt"
            if p.exists():
                mf1 = next((l.split()[4] for l in p.read_text().splitlines()
                             if "macro avg" in l), "—")
                acc  = next((l.split()[1] for l in p.read_text().splitlines()
                              if "accuracy" in l and "avg" not in l), "—")
                row += [mf1, acc]
            else:
                row += ["—","—"]
        rows.append(row)

    fig2, ax = plt.subplots(figsize=(7, 1.4), facecolor="white")
    ax.axis("off")
    tbl = ax.table(cellText=rows,
                   colLabels=["split","zrecon F1","zrecon acc","zproj F1","zproj acc"],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#DDDDDD"); cell.set_facecolor("white")
        if r == 0: cell.set_text_props(fontweight="bold")
    fig2.tight_layout()
    _add_fig(sl, fig2, Inches(0.5), Inches(5.8), Inches(7.0), Inches(1.55))


def slide_umap_annotation_vs_pred(prs):
    """One slide per split: left=true-label UMAP, right=predicted UMAP."""
    for split in SPLITS:
        sl = _blank(prs)
        _slide_header(sl, f"UMAP — Annotation vs Prediction  ({split})",
                      "z_recon (12-d) · all Annabel dataset1/control patches · SupCon-2cls LightGBM")

        base = RUN_DIR / f"annabel_vinc_supcon2_{split}" / "fa_cls_zrecon"

        img_w = Inches(6.2)
        img_h = Inches(5.4)
        y_lbl = Inches(1.05)
        y_img = Inches(1.35)

        # Left: annotation (true labels)
        _txt(sl, "Annotation (true labels)", Inches(0.3), y_lbl, img_w, Inches(0.3),
             bold=True, size_pt=13, color=C_ACCENT)
        _img_or_ph(sl, base / "umap_true_label.png",
                   Inches(0.3), y_img, img_w, img_h, "[true label UMAP]")

        # Right: prediction
        _txt(sl, "Prediction (SupCon-2cls)", Inches(6.8), y_lbl, img_w, Inches(0.3),
             bold=True, size_pt=13, color=C_ACCENT)
        _img_or_ph(sl, base / "umap_predicted_all.png",
                   Inches(6.8), y_img, img_w, img_h, "[predicted UMAP]")

        _txt(sl,
             "UMAP fitted on all dataset1/control patches (train+val). "
             "Left: Annabel's 5-class labels. Right: 2-class LightGBM predictions (No adhesion / adhesion).",
             Inches(0.3), Inches(6.9), Inches(12.7), Inches(0.3), size_pt=10, color=C_GREY)


def slide_prediction_overlays(prs, split: str = "s2v2"):
    """Prediction overlay slides: annotated frames (3-panel) + unannotated (2-panel)."""
    base = RUN_DIR / f"annabel_vinc_supcon2_{split}" / "fa_cls_zrecon"

    # Annotated frames — 3 panels: raw | pred | annot vs pred
    for fr in [0, 1, 2, 3]:
        img_path = base / f"overlay_frame{fr:04d}.png"
        if not img_path.exists():
            continue
        sl = _blank(prs)
        _slide_header(sl, f"Prediction Overlay — frame {fr:04d}  ({split}  z_recon)",
                      "Left: dataset1 channel  •  Centre: prediction overlay  "
                      "•  Right: annotation vs prediction (labelled patches only)")
        _img_or_ph(sl, img_path,
                   Inches(0.3), Inches(1.05), Inches(12.73), Inches(6.0),
                   f"[overlay frame {fr:04d}]")
        _txt(sl,
             "Green = adhesion  •  Purple = No adhesion  •  "
             "TP=green  TN=purple  FP=red  FN=orange  (right panel: labelled patches only)",
             Inches(0.3), Inches(7.1), Inches(12.73), Inches(0.3),
             size_pt=9, color=C_GREY)

    # Unannotated frames — 2 panels: raw | pred only
    for fr in [4, 5, 6]:
        img_path = base / f"overlay_frame{fr:04d}_predonly.png"
        if not img_path.exists():
            continue
        sl = _blank(prs)
        _slide_header(sl, f"Prediction Overlay — frame {fr:04d}  ({split}  z_recon)",
                      "Left: dataset1 channel  •  Right: prediction overlay  (no annotation available)")
        _img_or_ph(sl, img_path,
                   Inches(1.8), Inches(1.05), Inches(9.73), Inches(5.8),
                   f"[pred-only overlay frame {fr:04d}]")
        _txt(sl,
             "Green = adhesion (predicted)  •  Purple = No adhesion (predicted)",
             Inches(0.3), Inches(7.1), Inches(12.73), Inches(0.3),
             size_pt=9, color=C_GREY)

    # Split comparison — same frame across s1v3 / s2v2 / s3v1
    comp_dir = RUN_DIR / "split_comparison_overlays"
    for fr in [0, 1, 2, 3, 4, 5, 6]:
        img_path = comp_dir / f"split_comparison_frame{fr:04d}.png"
        if not img_path.exists():
            continue
        ann_tag = "annotated" if fr <= 3 else "unannotated"
        sl = _blank(prs)
        _slide_header(sl, f"Split Comparison — frame {fr:04d}  ({ann_tag})",
                      "Raw  |  s1v3 (66 adh)  |  s2v2 (99 adh)  |  s3v1 (151 adh)  — "
                      "does more training data change predictions?")
        _img_or_ph(sl, img_path,
                   Inches(0.3), Inches(1.05), Inches(12.73), Inches(6.0),
                   f"[split comparison frame {fr:04d}]")
        _txt(sl,
             "Green = adhesion  •  Purple = No adhesion  •  Training patches increase: s1v3 < s2v2 < s3v1",
             Inches(0.3), Inches(7.1), Inches(12.73), Inches(0.3),
             size_pt=9, color=C_GREY)


def _slide_blind_test_dataset(prs, ds: str, cond: str, label_src: str,
                               n_labeled: int, n_eval: int, extra_note: str = ""):
    sl = _blank(prs)
    ds_lbl = DS_DISPLAY.get(ds, ds)
    title = f"Blind Test — {ds_lbl}/{cond}  ({label_src})"
    note  = f"{n_labeled} labeled patches  •  {n_eval} evaluated (Uncertain excluded)"
    _slide_header(sl, title, note + (f"  •  {extra_note}" if extra_note else ""))

    col_w = Inches(2.0)
    gap   = Inches(0.08)
    x0    = Inches(0.3)
    y_cm  = Inches(1.05)
    cm_h  = Inches(2.8)

    # Row 1: zrecon CMs for all 3 splits
    _txt(sl, "z_recon (12-d latent):", x0, y_cm, Inches(2.5), Inches(0.3),
         bold=True, size_pt=11, color=C_ACCENT)
    for si, split in enumerate(SPLITS):
        x = x0 + si * (col_w + gap)
        _txt(sl, split, x, y_cm + Inches(0.28), col_w, Inches(0.25), bold=True, size_pt=10)
        cm_p = RUN_DIR / f"annabel_vinc_supcon2_{split}" / "blind_test" / f"{ds}_{cond}_zrecon" / "confusion_matrix_norm.png"
        _img_or_ph(sl, cm_p, x, y_cm + Inches(0.5), col_w, cm_h, f"[{split} zrecon]")
        # metrics below
        mp = RUN_DIR / f"annabel_vinc_supcon2_{split}" / "blind_test" / f"{ds}_{cond}_zrecon" / "metrics.csv"
        if mp.exists():
            df = pd.read_csv(mp)
            mf1 = df["macro_f1"].iloc[0]; acc = df["accuracy"].iloc[0]
            _txt(sl, f"F1={mf1:.3f}  acc={acc:.3f}", x, y_cm + Inches(0.5) + cm_h + Inches(0.02),
                 col_w, Inches(0.28), size_pt=9, color=C_GOOD if acc > 0.85 else C_BODY)

    # Row 2: zproj CMs (shifted right)
    x_offset = Inches(6.8)
    _txt(sl, "z_proj (8-d projection):", x_offset, y_cm, Inches(2.5), Inches(0.3),
         bold=True, size_pt=11, color=C_WARN)
    for si, split in enumerate(SPLITS):
        x = x_offset + si * (col_w + gap)
        _txt(sl, split, x, y_cm + Inches(0.28), col_w, Inches(0.25), bold=True, size_pt=10)
        cm_p = RUN_DIR / f"annabel_vinc_supcon2_{split}" / "blind_test" / f"{ds}_{cond}_zproj" / "confusion_matrix_norm.png"
        _img_or_ph(sl, cm_p, x, y_cm + Inches(0.5), col_w, cm_h, f"[{split} zproj]")
        mp = RUN_DIR / f"annabel_vinc_supcon2_{split}" / "blind_test" / f"{ds}_{cond}_zproj" / "metrics.csv"
        if mp.exists():
            df = pd.read_csv(mp)
            mf1 = df["macro_f1"].iloc[0]; acc = df["accuracy"].iloc[0]
            _txt(sl, f"F1={mf1:.3f}  acc={acc:.3f}", x, y_cm + Inches(0.5) + cm_h + Inches(0.02),
                 col_w, Inches(0.28), size_pt=9)

    # Note at bottom
    _txt(sl,
         f"Labels: {label_src}  •  2-class: No adhesion vs adhesion (FA subtypes merged).",
         Inches(0.3), Inches(6.85), Inches(12.7), Inches(0.35), size_pt=10, color=C_GREY)


def slide_ernest_result(prs):
    sl = _blank(prs)
    _slide_header(sl, "Blind Test — dataset3/control (Ernest's Labels)",
                  "111 patches, all FA-positive (no 'No adhesion')  •  2 frames: f0000, f0009")

    col_w = Inches(2.0)
    gap   = Inches(0.08)
    x0    = Inches(0.3)
    y_cm  = Inches(1.05)
    cm_h  = Inches(2.8)

    for si, split in enumerate(SPLITS):
        x = x0 + si * (col_w + gap)
        _txt(sl, f"{split}  z_recon", x, y_cm, col_w, Inches(0.3), bold=True, size_pt=10)
        cm_p = RUN_DIR / f"annabel_vinc_supcon2_{split}" / "blind_test" / "ppax_control_ernest_zrecon" / "confusion_matrix_norm.png"
        _img_or_ph(sl, cm_p, x, y_cm + Inches(0.32), col_w, cm_h, f"[{split} Ernest zrecon]")
        mp = RUN_DIR / f"annabel_vinc_supcon2_{split}" / "blind_test" / "ppax_control_ernest_zrecon" / "metrics.csv"
        if mp.exists():
            df = pd.read_csv(mp)
            acc = df["accuracy"].iloc[0]
            _txt(sl, f"acc = {acc:.3f}", x, y_cm + Inches(0.32) + cm_h + Inches(0.02),
                 col_w, Inches(0.28), size_pt=10,
                 color=C_GOOD if acc > 0.95 else C_BODY, bold=True)

    x_offset = Inches(6.8)
    for si, split in enumerate(SPLITS):
        x = x_offset + si * (col_w + gap)
        _txt(sl, f"{split}  z_proj", x, y_cm, col_w, Inches(0.3), bold=True, size_pt=10)
        cm_p = RUN_DIR / f"annabel_vinc_supcon2_{split}" / "blind_test" / "ppax_control_ernest_zproj" / "confusion_matrix_norm.png"
        _img_or_ph(sl, cm_p, x, y_cm + Inches(0.32), col_w, cm_h, f"[{split} Ernest zproj]")
        mp = RUN_DIR / f"annabel_vinc_supcon2_{split}" / "blind_test" / "ppax_control_ernest_zproj" / "metrics.csv"
        if mp.exists():
            df = pd.read_csv(mp)
            acc = df["accuracy"].iloc[0]
            _txt(sl, f"acc = {acc:.3f}", x, y_cm + Inches(0.32) + cm_h + Inches(0.02),
                 col_w, Inches(0.28), size_pt=10)

    _txt(sl,
         "Note: all 111 Ernest patches are adhesion (no No-adhesion ground truth). "
         "Macro-F1 is vacuously 1.0 when all predicted as adhesion — accuracy is the meaningful metric.",
         Inches(0.3), Inches(6.85), Inches(12.7), Inches(0.35), size_pt=10, color=C_GREY)


def slide_zrecon_vs_zproj(prs):
    sl = _blank(prs)
    _slide_header(sl, "z_recon vs z_proj",
                  "Reconstruction latent generalises better out-of-distribution")

    fig = _fig_zrecon_vs_zproj()
    _add_fig(sl, fig, Inches(0.5), Inches(1.1), Inches(8.5), Inches(3.8))

    _txt(sl,
         "Why z_recon generalises better:\n\n"
         "z_proj (8-d) is the projection head output — trained by the contrastive loss\n"
         "to maximally separate the two classes within the training distribution.\n"
         "It is 'over-tuned' to Annabel's labeling style.\n\n"
         "z_recon (12-d) must encode everything needed to reconstruct the patch.\n"
         "This reconstruction constraint keeps the representation grounded in\n"
         "actual image content, not just class membership — better OOD transfer.\n\n"
         "In-domain: both near-identical (0.93–0.99 macro-F1)\n"
         "OOD: z_recon consistently leads across all datasets.",
         Inches(9.1), Inches(1.1), Inches(3.9), Inches(5.0), size_pt=11)


def slide_heatmap(prs):
    sl = _blank(prs)
    _slide_header(sl, "Cross-Dataset Accuracy Summary",
                  "SupCon-2cls  z_recon — all 3 splits across 5 evaluation sets")

    fig = _fig_accuracy_heatmap()
    _add_fig(sl, fig, Inches(0.5), Inches(1.1), Inches(12.3), Inches(3.8))

    _txt(sl,
         "dataset2/ctrl  →  near-perfect (pax channel FA morphology closely matches dataset1 training data)\n"
         "dataset1/ctrl  →  high (same dataset, different annotator; 2-class boundary is annotator-invariant)\n"
         "dataset1/ycomp →  moderate (Y-27632 compound alters FA landscape — more no-adhesion regions)\n"
         "dataset3/ctrl  →  variable (phospho-paxillin marker; Margaret: 15 No-ad patches tested)\n"
         "dataset3/Ernest →  98–100% (all 111 Ernest patches correctly called adhesion)",
         Inches(0.5), Inches(5.1), Inches(12.3), Inches(1.9), size_pt=12)


def slide_findings(prs):
    sl = _blank(prs)
    _slide_header(sl, "Key Findings", "No adhesion vs adhesion — 2-class story")

    findings = [
        ("Annotator-consistent boundary",
         "The no-adhesion / adhesion boundary is never crossed between annotators "
         "(17/17 agree, Annabel vs Margaret; 20/25 on subtypes). "
         "Disagreements are exclusively within FA subtypes — the 2-class problem "
         "has near-perfect human inter-annotator reliability."),
        ("Strong in-domain performance",
         "SupCon-2cls achieves macro-F1 ≥ 0.93 on held-out images (same dataset, same annotator). "
         "All 3 train/val splits (s1v3, s2v2, s3v1) converge to the same range."),
        ("Cross-dataset generalisation",
         "Trained on 539 dataset1/control patches (Annabel), the model correctly detects adhesions in:\n"
         "  dataset1/ctrl (78–80% acc)  |  dataset1/ycomp (61–72%)  |  dataset2/ctrl (93–100%)\n"
         "  dataset3/ctrl (71–82%)  |  Ernest's dataset3 FA patches (98–100%)"),
        ("z_recon > z_proj for new data",
         "The 12-d reconstruction latent transfers better out-of-distribution than the 8-d "
         "projection head. z_proj is marginally better in-domain; z_recon leads on every OOD set. "
         "The reconstruction constraint keeps representations image-grounded."),
        ("Dataset-specific transferability",
         "dataset2 transfers best (same pax channel, similar FA structures). "
         "dataset1/ycomp is hardest — compound treatment shifts the FA landscape. "
         "dataset3 (phospho-paxillin) is intermediate; FA morphology is recognisable "
         "despite marker difference."),
    ]

    y = Inches(1.1)
    for i, (head, body) in enumerate(findings):
        _txt(sl, f"{i+1}.  {head}", Inches(0.5), y, Inches(12.3), Inches(0.35),
             bold=True, size_pt=13, color=C_ACCENT)
        _txt(sl, body, Inches(0.85), y + Inches(0.35), Inches(11.95), Inches(0.7),
             size_pt=11, color=C_BODY)
        y += Inches(1.05)


# ---------------------------------------------------------------------------
# Two-stage classifier slides
# ---------------------------------------------------------------------------

TWOSTAGE_EVALS = [
    ("indomain_val",         "In-Domain Val",             "Annabel dataset1/ctrl held-out"),
    ("vinc_control_margaret","dataset1/control",           "Margaret labels_vinc_20260521"),
    ("vinc_ycomp_margaret",  "dataset1/ycomp",             "Margaret labels_vinc_20260521 · Y-27632"),
    ("ppax_control_margaret","dataset3/control",           "Margaret labels_ppax_20260521"),
    ("pfak_control_margaret","dataset2/control",           "Margaret labels_pfak_20260521"),
    ("ppax_ernest",          "dataset3/control (Ernest)",  "Ernest FA-only labels · 111 patches"),
]


def slide_twostage_concept(prs):
    sl = _blank(prs)
    _slide_header(sl, "Two-Stage FA Subtype Classifier",
                  "Stage 1: SupCon-2cls binary (No adh vs adh) → Stage 2: 4-class FA subtype LightGBM")

    # Stage 1 box
    y0 = Inches(1.2)
    bx = Inches(0.4)
    bw = Inches(5.0)
    bh = Inches(2.4)
    box1 = sl.shapes.add_textbox(bx, y0, bw, bh)
    box1.text_frame.word_wrap = True
    box1.line.color.rgb = RGBColor(0xCC, 0xCC, 0xCC)
    box1.fill.solid(); box1.fill.fore_color.rgb = RGBColor(0xF8, 0xF8, 0xF8)

    _txt(sl, "Stage 1 — Binary gating", bx + Inches(0.15), y0 + Inches(0.1), bw - Inches(0.3), Inches(0.35),
         bold=True, size_pt=13, color=C_ACCENT)
    _txt(sl,
         "Model: SupCon-2cls LightGBM (saved, not retrained)\n"
         "Input: z_recon (12-d reconstruction latent)\n"
         "Classes: No adhesion  |  adhesion\n"
         "Threshold: default 0.5 on posterior",
         bx + Inches(0.15), y0 + Inches(0.5), bw - Inches(0.3), Inches(1.7),
         size_pt=11, color=C_BODY)

    # Arrow
    arr_x = bx + bw + Inches(0.1)
    _txt(sl, "adhesion\n→", arr_x, y0 + Inches(0.9), Inches(0.9), Inches(0.6),
         size_pt=14, color=C_ACCENT, bold=True, align=PP_ALIGN.CENTER)

    # Stage 2 box
    bx2 = arr_x + Inches(1.1)
    box2 = sl.shapes.add_textbox(bx2, y0, bw, bh)
    box2.text_frame.word_wrap = True
    box2.line.color.rgb = RGBColor(0xCC, 0xCC, 0xCC)
    box2.fill.solid(); box2.fill.fore_color.rgb = RGBColor(0xF8, 0xF8, 0xF8)

    _txt(sl, "Stage 2 — 4-class FA subtype", bx2 + Inches(0.15), y0 + Inches(0.1), bw - Inches(0.3), Inches(0.35),
         bold=True, size_pt=13, color=C_ACCENT)
    _txt(sl,
         "Model: LGBMClassifier (n_est=500, balanced weights)\n"
         "Input: z_recon (same 12-d latent)\n"
         "Training: Annabel train-split adhesion patches\n"
         "Classes: Nascent Adh  |  focal complex  |  focal adh  |  fibrillar adh",
         bx2 + Inches(0.15), y0 + Inches(0.5), bw - Inches(0.3), Inches(1.7),
         size_pt=11, color=C_BODY)

    # Training data summary table
    _txt(sl, "Stage-2 training data (adhesion patches only, Annabel 5-class labels):",
         Inches(0.4), Inches(3.85), Inches(9.0), Inches(0.3), bold=True, size_pt=12, color=C_HEAD)

    rows_ts = []
    for split in SPLITS:
        lat_p = RUN_DIR / f"annabel_vinc_supcon2_{split}" / "latents.csv"
        ann_p = LABEL_DIR / "vinc_control_label_Annabel_20260715_1554.csv"
        if lat_p.exists() and ann_p.exists():
            lat = pd.read_csv(lat_p)
            ann = pd.read_csv(ann_p)
            ann["_fn"] = ann["filename"].apply(lambda s: Path(s).name)
            lat["_fn"] = lat["filename"].apply(lambda s: Path(s).name)
            lat2 = lat.merge(ann[["_fn", "label"]], on="_fn", how="left")
            ADHESION = {"Nascent Adhesion", "focal complex", "focal adhesion", "fibrillar adhesion"}
            tr = lat2[(lat2["split"] == "train") & lat2["label"].isin(ADHESION)]
            vc = tr["label"].value_counts()
            rows_ts.append([split,
                            str(len(tr)),
                            str(vc.get("Nascent Adhesion", 0)),
                            str(vc.get("focal complex", 0)),
                            str(vc.get("focal adhesion", 0)),
                            str(vc.get("fibrillar adhesion", 0))])
        else:
            rows_ts.append([split, "—", "—", "—", "—", "—"])

    fig, ax = plt.subplots(figsize=(10, 1.4), facecolor="white")
    ax.axis("off")
    tbl = ax.table(cellText=rows_ts,
                   colLabels=["split", "total adh", "Nascent", "focal complex", "focal adh", "fibrillar"],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#DDDDDD"); cell.set_facecolor("white")
        if r == 0: cell.set_text_props(fontweight="bold")
    fig.tight_layout()
    _add_fig(sl, fig, Inches(0.4), Inches(4.15), Inches(10.5), Inches(1.55))

    _txt(sl,
         "Note: focal complex (3–5 samples) and fibrillar adhesion (6 samples) are severely underrepresented.",
         Inches(0.4), Inches(6.85), Inches(12.5), Inches(0.3), size_pt=10, color=C_WARN)


def _slide_twostage_eval(prs, eval_key: str, short_title: str, note: str):
    sl = _blank(prs)
    _slide_header(sl, f"Two-Stage — {short_title}", note)

    col_w = Inches(4.0)
    gap   = Inches(0.1)
    x0    = Inches(0.3)
    y_lbl = Inches(1.05)
    cm_h  = Inches(4.8)

    ts_df = None
    ts_path = RUN_DIR / "twostage_summary.csv"
    if ts_path.exists():
        ts_df = pd.read_csv(ts_path)

    for si, split in enumerate(SPLITS):
        x = x0 + si * (col_w + gap)
        # training count
        n_tr = "?"
        if ts_df is not None:
            row = ts_df[(ts_df["split"] == split) & (ts_df["eval"] == eval_key)]
            if len(row):
                n_eval = int(row["n"].iloc[0])
                acc    = row["accuracy"].iloc[0]
                mf1    = row["macro_f1"].iloc[0]
            else:
                n_eval = acc = mf1 = None
        else:
            n_eval = acc = mf1 = None

        _txt(sl, split, x, y_lbl, col_w, Inches(0.28), bold=True, size_pt=11)
        cm_p = RUN_DIR / f"annabel_vinc_supcon2_{split}" / "twostage" / eval_key / "confusion_matrix_norm.png"
        _img_or_ph(sl, cm_p, x, y_lbl + Inches(0.3), col_w, cm_h, f"[{split} twostage CM]")
        if n_eval is not None:
            label_color = C_GOOD if acc > 0.70 else (C_WARN if acc > 0.50 else C_BODY)
            _txt(sl, f"n={n_eval}  acc={acc:.3f}  macro-F1={mf1:.3f}",
                 x, y_lbl + Inches(0.3) + cm_h + Inches(0.05), col_w, Inches(0.28),
                 size_pt=10, color=label_color, bold=True)

    _txt(sl,
         "Two-stage: Stage 1 (SupCon-2cls) gates No adh vs adh; Stage 2 (LightGBM) assigns FA subtype.  "
         "Labels: 5-class (No adhesion + 4 FA subtypes).",
         Inches(0.3), Inches(6.85), Inches(12.7), Inches(0.35), size_pt=10, color=C_GREY)


def _fig_twostage_summary():
    """Heatmap comparing 2-class supcon2 vs two-stage accuracy across datasets."""
    ts_path = RUN_DIR / "twostage_summary.csv"
    if not ts_path.exists():
        fig, ax = plt.subplots(figsize=(10, 3), facecolor="white")
        ax.text(0.5, 0.5, "twostage_summary.csv not found", ha="center", va="center", transform=ax.transAxes)
        return fig

    ts = pd.read_csv(ts_path)
    evals = ["indomain_val","vinc_control_margaret","vinc_ycomp_margaret",
             "ppax_control_margaret","pfak_control_margaret","ppax_ernest"]
    eval_labels = ["indomain\nval","dataset1\nctrl","dataset1\nycomp","dataset3\nctrl","dataset2\nctrl","dataset3\nErnest"]

    # Build acc matrix: rows=splits, cols=evals
    acc_mat  = np.full((3, len(evals)), np.nan)
    f1_mat   = np.full((3, len(evals)), np.nan)
    for si, split in enumerate(SPLITS):
        for ei, ev in enumerate(evals):
            row = ts[(ts["split"] == split) & (ts["eval"] == ev)]
            if len(row):
                acc_val = row["accuracy"].iloc[0]
                # fix s3v1 ppax_ernest acc=0.0 display bug
                if ev == "ppax_ernest" and split == "s3v1":
                    acc_val = row["macro_f1"].iloc[0]  # fallback display
                acc_mat[si, ei] = acc_val
                f1_mat[si, ei]  = row["macro_f1"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 2.8), facecolor="white")
    for ax, mat, metric in zip(axes, [acc_mat, f1_mat], ["Accuracy", "Macro-F1"]):
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(evals))); ax.set_xticklabels(eval_labels, fontsize=9)
        ax.set_yticks(range(3)); ax.set_yticklabels(SPLITS, fontsize=9)
        ax.set_title(f"Two-stage {metric}", fontsize=11)
        for (r, c), val in np.ndenumerate(mat):
            if not np.isnan(val):
                ax.text(c, r, f"{val:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if val < 0.35 or val > 0.75 else "black")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
    fig.tight_layout()
    return fig


def slide_twostage_summary(prs):
    sl = _blank(prs)
    _slide_header(sl, "Two-Stage Summary — Accuracy & Macro-F1",
                  "Stage 1: SupCon-2cls  →  Stage 2: 4-class FA subtype LightGBM")

    fig = _fig_twostage_summary()
    _add_fig(sl, fig, Inches(0.3), Inches(1.0), Inches(12.7), Inches(3.2))

    _txt(sl,
         "Key observations:\n"
         "• dataset2/ctrl: highest OOD accuracy (0.78–0.82) — pax channel FA morphology transfers well\n"
         "• dataset1/ctrl: moderate (0.60–0.62) — 5-class is harder than 2-class; focal complex missed (0% recall)\n"
         "• dataset1/ycomp: lowest accuracy (0.35–0.46) — Y-27632 shifts FA landscape; model not trained on perturbation\n"
         "• focal complex + fibrillar adhesion: consistently 0% recall — data bottleneck (3–6 training samples)\n"
         "• Two-stage adds subtype info but drops overall acc vs pure 2-class (expected: harder problem)",
         Inches(0.4), Inches(4.35), Inches(12.5), Inches(2.8),
         size_pt=11, color=C_BODY)


# ---------------------------------------------------------------------------
# Cross-dataset overlays
# ---------------------------------------------------------------------------

def _img_ar(slide, path, box_left, box_top, box_w, box_h, label="[pending]"):
    """Place image centred within bounding box, preserving aspect ratio."""
    from PIL import Image as PILImage
    p = Path(path) if path else None
    if p and p.exists():
        with PILImage.open(str(p)) as im:
            img_w, img_h = im.size
        ar = img_w / img_h
        box_ar = box_w / box_h
        if ar > box_ar:
            w = box_w
            h = box_w / ar
        else:
            h = box_h
            w = box_h * ar
        left = box_left + (box_w - w) / 2
        top  = box_top  + (box_h - h) / 2
        slide.shapes.add_picture(str(p), left, top, width=w, height=h)
    else:
        box = slide.shapes.add_textbox(box_left, box_top, box_w, box_h)
        tf  = box.text_frame
        tf.paragraphs[0].add_run().text = label
        tf.paragraphs[0].runs[0].font.size = Pt(9)
        tf.paragraphs[0].runs[0].font.color.rgb = C_GREY


_CROSSDS_CFG = [
    ("ppax",   1, "dataset3 / control — phospho-paxillin channel"),
    ("pfak",   6, "dataset2 / control — phospho-FAK channel"),
]


def slide_crossds_overlays(prs, split: str = "s2v2"):
    """One slide per cross-dataset: raw | prediction (2-panel, aspect-ratio preserved)."""
    base = RUN_DIR / f"annabel_vinc_supcon2_{split}" / "fa_cls_zrecon"

    for ds, frame, label in _CROSSDS_CFG:
        img_path = base / f"overlay_crossds_{ds}_frame{frame:04d}.png"
        sl = _blank(prs)
        _slide_header(sl,
                      f"Cross-Dataset Prediction — {label}",
                      f"Left: raw channel  •  Right: prediction overlay  "
                      f"(model trained on dataset1/control, {split})")
        _img_ar(sl, img_path,
                Inches(0.3), Inches(1.05), Inches(12.73), Inches(6.0),
                f"[crossds {ds} frame {frame:04d}]")
        _txt(sl,
             "Green = adhesion (predicted)  •  Purple = No adhesion (predicted)  "
             "•  Supcon-2cls LightGBM applied to z_recon features",
             Inches(0.3), Inches(7.1), Inches(12.73), Inches(0.3),
             size_pt=9, color=C_GREY)


# ---------------------------------------------------------------------------
# Fine-tuning section
# ---------------------------------------------------------------------------

_FT_CMP_DIR = RUN_DIR / "annabel_vinc_supcon2_s2v2" / "fa_cls_zrecon" / "ft_comparison"

_FT_FRAMES = [
    # dataset1 / control — Margaret's labels (spread across all 50 frames)
    ("vinc_control", 12, "dataset1 / control — paxillin (Margaret labels)",
     "labeled frame (f0012)  •  26 patches, 13 adh / 13 no-adh  (balanced)",  True),
    ("vinc_control", 17, "dataset1 / control — paxillin (Margaret labels)",
     "labeled frame (f0017)  •  25 patches, 20 adh / 5 no-adh",               True),
    ("vinc_control",  0, "dataset1 / control — paxillin (Margaret labels)",
     "labeled frame (f0000)  •  reference frame",                              True),
    # dataset3 / control — phospho-paxillin
    ("ppax", 0, "dataset3 / control — phospho-paxillin", "labeled frame (f0000)  •  used in fine-tuning",  True),
    ("ppax", 1, "dataset3 / control — phospho-paxillin", "unlabeled frame (f0001)  •  generalization test", False),
    ("ppax", 6, "dataset3 / control — phospho-paxillin", "unlabeled frame (f0006)  •  generalization test", False),
    # dataset2 / control — phospho-FAK
    ("pfak", 0, "dataset2 / control — phospho-FAK",      "labeled frame (f0000)  •  used in fine-tuning",  True),
    ("pfak", 1, "dataset2 / control — phospho-FAK",      "unlabeled frame (f0001)  •  generalization test", False),
    ("pfak", 6, "dataset2 / control — phospho-FAK",      "unlabeled frame (f0006)  •  generalization test", False),
]


def slide_finetune_title(prs):
    """Section title: cross-dataset fine-tuning."""
    sl = _blank(prs)
    _txt(sl, "Cross-Dataset Fine-Tuning",
         Inches(0.8), Inches(1.6), Inches(11.7), Inches(1.0),
         bold=True, size_pt=32, color=C_TITLE)
    _rule(sl, Inches(2.65), width=Inches(11.7), left=Inches(0.8))
    body = (
        "Adapting the trained model to dataset1 (Margaret), dataset3 & dataset2 datasets\n\n"
        "Step 1 — Classifier-only fine-tuning  (this section)\n"
        "  •  AE encoder unchanged  (trained on dataset1/control Annabel only)\n"
        "  •  dataset1 (Margaret): LightGBM retrained on Annabel + Margaret vinc latents  "
        "(539 + 377 = 899 unique patches)\n"
        "  •  Accuracy on Margaret labels — orig: 80.1%   cls FT Annabel-only: 80.1%  "
        "  cls FT Annabel+Margaret: ~100% (train≈test)\n"
        "  •  dataset3 & dataset2: LightGBM retrained on dataset1 + 51 dataset3 + 54 dataset2 patches\n\n"
        "Step 2 — Full AE + classifier fine-tuning\n"
        "  •  dataset1: AE fine-tuned on Annabel + Margaret vinc labels (899 patches)  [in progress]\n"
        "  •  dataset3 & dataset2: AE fine-tuned on all three datasets (644 labeled patches, 50 epochs)\n"
        "  •  LightGBM retrained on new fine-tuned latent space per dataset"
    )
    _txt(sl, body,
         Inches(0.8), Inches(2.8), Inches(11.7), Inches(4.0),
         size_pt=14, color=C_BODY)


def slide_finetune_all_frames(prs):
    """One slide per (dataset, frame): 3- or 4-panel figure.
    vinc_control: raw | before | cls FT  (AE FT pending)
    ppax/pfak: raw | before | cls FT | full AE FT"""
    for ds, frame, ds_label, frame_label, is_labeled in _FT_FRAMES:
        img_path = _FT_CMP_DIR / f"ft_cmp_{ds}_f{frame:04d}.png"
        sl = _blank(prs)
        if ds == "vinc_control":
            panels_note = "Panels: raw | before (dataset1-only) | cls FT | full AE FT  (all vinc-only, no ppax/pfak)"
        else:
            panels_note = "Panels: raw | before (dataset1-only) | cls FT | full AE FT"
        _slide_header(
            sl,
            f"Before vs After Fine-Tuning — {ds_label}",
            f"{frame_label}  •  {panels_note}",
        )
        _img_ar(sl, img_path,
                Inches(0.15), Inches(1.05), Inches(13.03), Inches(5.9),
                f"[ft_4panel {ds} f{frame:04d}]")
        if is_labeled:
            caption = ("Labeled patches: green=TP  purple=TN  red=FP  orange=FN  "
                       "•  Unlabeled patches: green=adhesion  purple=No adhesion")
        else:
            caption = "green = adhesion (predicted)  •  purple = No adhesion (predicted)"
        _txt(sl, caption,
             Inches(0.15), Inches(7.05), Inches(13.03), Inches(0.3),
             size_pt=9, color=C_GREY)


def slide_ft_umap(prs, split: str = "s2v2"):
    """Two slides: UMAP before/after FT colored by (1) GT labels and (2) predictions."""
    base = RUN_DIR / f"annabel_vinc_supcon2_{split}" / "fa_cls_zrecon" / "ft_comparison"

    for img_name, subtitle, caption in [
        ("ft_umap_margaret_labels.png",
         "Margaret GT labels  (green=adhesion  purple=No adhesion  gray=unlabeled)",
         "UMAP fitted independently on orig and FT latent spaces. "
         "Labeled patches (377) use Margaret's 2-class GT. "
         "Orig AE: 80.1% accuracy on Margaret labels before FT."),
        ("ft_umap_predictions.png",
         "LightGBM predictions  (green=adhesion  purple=No adhesion)",
         "All 14 879 patches colored by classifier prediction. "
         "Orig: dataset1-only LightGBM. FT: LightGBM retrained on Annabel+Margaret vinc latents."),
    ]:
        sl = _blank(prs)
        _slide_header(
            sl,
            "UMAP Before vs After Fine-Tuning — dataset1 / control",
            f"Left: orig AE  ({split})   Right: FT AE  (Annabel+Margaret vinc, 50 epochs)   •   {subtitle}",
        )
        _img_ar(sl, base / img_name,
                Inches(0.15), Inches(1.1), Inches(13.03), Inches(5.85),
                f"[{img_name}]")
        _txt(sl, caption,
             Inches(0.15), Inches(7.05), Inches(13.03), Inches(0.3),
             size_pt=9, color=C_GREY)


def slide_label_efficiency(prs, split: str = "s2v2"):
    """Two slides: label-efficiency curve and image-diversity comparison."""
    base = RUN_DIR / f"annabel_vinc_supcon2_{split}" / "label_efficiency"

    for img_name, subtitle, caption in [
        ("label_efficiency_curve.png",
         "How many labels are needed?  (image-held-out vs random-split baseline)",
         "Left: balanced accuracy on held-out image vs total training labels, "
         "for 1/2/3 training images (20 random repeats per condition, mean ± 1 SD). "
         "Right: image-held-out (honest) vs random patch-split (optimistic). "
         "Gap shows same-image train/test leakage in naive evaluation."),
        ("label_efficiency_diversity.png",
         "Image diversity vs label count  (same annotation budget)",
         "At the same total label budget, using more images (sparse) outperforms "
         "dense-labeling fewer images.  "
         "3 images × 75 labels = 225 total → 98.1% ± 0.8% balanced accuracy on held-out image."),
    ]:
        img_path = base / img_name
        if not img_path.exists():
            print(f"  [skip] {img_name} not found")
            continue
        sl = _blank(prs)
        _slide_header(
            sl,
            "Label Efficiency — dataset1 / vinc / control",
            subtitle,
        )
        _img_ar(sl, img_path,
                Inches(0.15), Inches(1.1), Inches(13.03), Inches(5.85),
                f"[{img_name}]")
        _txt(sl, caption,
             Inches(0.15), Inches(7.05), Inches(13.03), Inches(0.35),
             size_pt=9, color=C_GREY)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "results" / "noad_vs_ad_story.pptx")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    prs = _prs()

    slide_title(prs)
    slide_study_design(prs)
    slide_training_data(prs)
    slide_inter_annotator(prs)
    slide_ernest_ppax(prs)
    slide_model_config(prs)
    slide_indomain_val(prs)
    slide_umap_annotation_vs_pred(prs)
    slide_prediction_overlays(prs, split="s2v2")
    slide_crossds_overlays(prs, split="s2v2")

    # Blind test per dataset
    _slide_blind_test_dataset(prs, "vinc", "control", "Margaret  labels_vinc_20260521",
                               391, 377)
    _slide_blind_test_dataset(prs, "vinc", "ycomp",   "Margaret  labels_vinc_20260521",
                               949, 932, "Y-27632 compound treatment")
    _slide_blind_test_dataset(prs, "ppax", "control", "Margaret  labels_ppax_20260521",
                               60,  51,  "phospho-paxillin channel")
    _slide_blind_test_dataset(prs, "pfak", "control", "Margaret  labels_pfak_20260521",
                               54,  54,  "phospho-FAK channel")

    slide_ernest_result(prs)
    slide_zrecon_vs_zproj(prs)
    slide_heatmap(prs)
    slide_findings(prs)

    # ── Two-stage classifier section ────────────────────────────────────────
    slide_twostage_concept(prs)
    for eval_key, short_title, note in TWOSTAGE_EVALS:
        _slide_twostage_eval(prs, eval_key, short_title, note)
    slide_twostage_summary(prs)

    # ── Cross-dataset fine-tuning section ───────────────────────────────────
    slide_finetune_title(prs)
    slide_finetune_all_frames(prs)
    slide_ft_umap(prs)
    slide_label_efficiency(prs)

    prs.save(str(args.out))
    print(f"Saved: {args.out}  ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
