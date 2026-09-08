#!/usr/bin/env python3
"""
make_pptx_ms_b2_ds1_ps64.py

PPT summarising the multiscale B2 DS1 ps=64 5-fold CV experiment.

Slides:
  1. Dataset overview — fold breakdown, class distribution
  2. Example patches — side-by-side ps=32 vs ps=64 for No adhesion and adhesion
  3. Experiment setup — comparison table of the 4 configurations
  4. Classification results — 5-fold balanced accuracy (ps64 lat64p32 done;
                               others filled in when available)
  5. Reconstruction examples — input vs recon for the example patches (ps=64)
  6. UMAP — all 1224 patches encoded from held-out fold, coloured by label / condition

Usage:
    python scripts/make_pptx_ms_b2_ds1_ps64.py
"""
from __future__ import annotations

import io
import sys
import random as _random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

REPO = Path(__file__).resolve().parents[1]
DATA = Path("/net/projects/CLS/lding/data/fa_data_analysis")
sys.path.insert(0, str(REPO))

RUN_TAG_PS64  = "ms_b2_ds1_ps64_lat64p32"
FRAME_BASE    = DATA / "ae_results" / "source_frames" / "cio_mode_prt" / "vinc"
LAB_DIR       = DATA / "labelling" / RUN_TAG_PS64
RUN_ROOT_PS64 = DATA / "ae_results" / "multiscale" / RUN_TAG_PS64
OUT           = REPO / "results" / "ms_b2_ds1_ps64.pptx"

SLIDE_W, SLIDE_H = 13.33, 7.5
N_FOLDS = 5

LABEL_COLORS = {"No adhesion": "#9467bd", "adhesion": "#2ca02c"}
COND_COLORS  = {"control": "#1f77b4", "ycomp": "#ff7f0e"}

# ── PPT helpers (same as make_pptx_label_overview.py) ────────────────────────

def _hex2rgb(h: str) -> RGBColor:
    h = h.lstrip("#")
    return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

def _paste_pil(slide, img, left, top, width, height):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    slide.shapes.add_picture(buf, Inches(left), Inches(top), Inches(width), Inches(height))

def _fig_to_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)

def _txt(slide, text, left, top, width, height,
         size=11, bold=False, color="#333333", align=PP_ALIGN.LEFT, italic=False):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.color.rgb = _hex2rgb(color)

def _rect(slide, left, top, width, height, fill_hex):
    shape = slide.shapes.add_shape(1,
        Inches(left), Inches(top), Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = _hex2rgb(fill_hex)
    shape.line.fill.background()

def _header(slide, title_text, subtitle=None):
    _rect(slide, 0, 0, SLIDE_W, 0.12, "#2c3e50")
    _txt(slide, title_text, 0.4, 0.18, SLIDE_W - 0.8, 0.55,
         size=20, bold=True, color="#2c3e50")
    if subtitle:
        _txt(slide, subtitle, 0.4, 0.72, SLIDE_W - 0.8, 0.38,
             size=10, color="#666666")

# ── Patch loading from source frames ─────────────────────────────────────────

def _load_frame(condition_name: str, frame_idx: int, channel: str = "pax") -> np.ndarray:
    path = FRAME_BASE / condition_name / f"{condition_name}_f{frame_idx:04d}_{channel}.tif"
    import tifffile
    return tifffile.imread(str(path)).astype(np.float32)

_frame_cache: dict = {}

def _get_frame(condition_name: str, frame_idx: int) -> np.ndarray:
    key = (condition_name, frame_idx)
    if key not in _frame_cache:
        _frame_cache[key] = _load_frame(condition_name, frame_idx)
    return _frame_cache[key]

def _crop_patch(condition_name: str, frame_idx: int, cx: int, cy: int,
                patch_size: int) -> np.ndarray:
    frame = _get_frame(condition_name, frame_idx)
    H, W  = frame.shape
    half  = patch_size // 2
    # reflect-pad
    pad   = half + 4
    padded = np.pad(frame, pad, mode="reflect")
    r0 = cy + pad - half
    c0 = cx + pad - half
    patch = padded[r0:r0 + patch_size, c0:c0 + patch_size]
    return patch

def _normalise_patch(patch: np.ndarray) -> np.ndarray:
    p2, p98 = np.percentile(patch, 2), np.percentile(patch, 98)
    return np.clip((patch - p2) / max(p98 - p2, 1e-8), 0, 1)

def _patch_pil(condition_name, frame_idx, cx, cy, ps) -> Image.Image:
    raw  = _crop_patch(condition_name, frame_idx, cx, cy, ps)
    norm = _normalise_patch(raw)
    scale = max(1, 128 // ps)
    img  = Image.fromarray((norm * 255).astype(np.uint8))
    return img.resize((ps * scale, ps * scale), Image.NEAREST)

# ── Pick example patches ──────────────────────────────────────────────────────

def _pick_examples(fs: pd.DataFrame, n_per_class: int = 6, seed: int = 42) -> pd.DataFrame:
    """Return n_per_class labeled rows per class (No adhesion / adhesion)."""
    rng = _random.Random(seed)
    rows = []
    for lbl in ["No adhesion", "adhesion"]:
        pool = fs[fs["label"] == lbl].to_dict("records")
        rng.shuffle(pool)
        rows.extend(pool[:n_per_class])
    return pd.DataFrame(rows)

# ── Slides ────────────────────────────────────────────────────────────────────

def _slide_dataset(prs, fs: pd.DataFrame):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _header(slide, "Dataset: B2 DS1 — 5-fold CV Setup",
            subtitle=("1,224 patches · control (539) + ycomp (685) · "
                      "Annabel labels · 2-class: No adhesion / adhesion · "
                      "patch-level random fold split (seed=42)"))

    # ── Left: fold table ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.axis("off")
    headers = ["Fold", "Train", "Test", "Train\nNo adh", "Train\nadh",
               "Test\nNo adh", "Test\nadh"]
    rows_data = []
    for k in range(N_FOLDS):
        tr = fs[fs["fold"] != k]
        te = fs[fs["fold"] == k]
        rows_data.append([
            f"fv{k}",
            len(tr), len(te),
            (tr["label"] == "No adhesion").sum(),
            (tr["label"] == "adhesion").sum(),
            (te["label"] == "No adhesion").sum(),
            (te["label"] == "adhesion").sum(),
        ])
    tbl = ax.table(cellText=rows_data, colLabels=headers,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f0f4f8")
    ax.set_title("Per-fold patch counts", fontsize=11, fontweight="bold", pad=8)
    fig.tight_layout()
    _paste_pil(slide, _fig_to_pil(fig), 0.3, 1.0, 6.5, 4.2)

    # ── Right: stacked bar — class × condition ────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(5.0, 3.5))
    groups = [("control", "No adhesion"), ("control", "adhesion"),
              ("ycomp",   "No adhesion"), ("ycomp",   "adhesion")]
    labels  = ["ctrl\nNo adh", "ctrl\nadh", "ycomp\nNo adh", "ycomp\nadh"]
    counts  = [len(fs[(fs.condition_name == c) & (fs.label == l)]) for c, l in groups]
    colors  = ["#9467bd", "#2ca02c", "#ff9f45", "#1a8a2c"]
    bars = ax2.bar(labels, counts, color=colors, edgecolor="white", linewidth=0.8)
    for bar, cnt in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 str(cnt), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Patch count", fontsize=10)
    ax2.set_title("Class × condition distribution", fontsize=11, fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)
    fig2.tight_layout()
    _paste_pil(slide, _fig_to_pil(fig2), 7.0, 1.0, 5.9, 4.2)

    _txt(slide, f"All 4 ctrl frames · 14 ycomp frames · patch-level random split "
         f"(not frame-based — patches from same frame may appear in both train and test)",
         0.3, 5.4, SLIDE_W - 0.6, 0.6, size=9, color="#888888", italic=True)


def _slide_example_patches(prs, examples: pd.DataFrame):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _header(slide, "Example Labeled Patches — ps=32 vs ps=64",
            subtitle="Top row: ps=32 crop · Bottom row: ps=64 crop · "
                     "Same center coordinates · Left: No adhesion · Right: adhesion")

    n_per_class = len(examples) // 2
    no_adh = examples[examples["label"] == "No adhesion"].head(n_per_class)
    adh    = examples[examples["label"] == "adhesion"].head(n_per_class)

    def _make_panel(subset: pd.DataFrame, title: str, color: str) -> Image.Image:
        n = len(subset)
        fig, axes = plt.subplots(2, n, figsize=(n * 1.4, 3.2))
        if n == 1:
            axes = axes[:, np.newaxis]
        fig.suptitle(title, fontsize=11, fontweight="bold", color=color, y=1.02)
        for col, (_, row) in enumerate(subset.iterrows()):
            for r_idx, ps in enumerate([32, 64]):
                img = _patch_pil(row["condition_name"], int(row["frame_idx"]),
                                 int(row["cx"]), int(row["cy"]), ps)
                axes[r_idx, col].imshow(np.array(img), cmap="gray", vmin=0, vmax=255)
                axes[r_idx, col].axis("off")
                if col == 0:
                    axes[r_idx, col].set_ylabel(f"ps={ps}", fontsize=8, labelpad=2)
        fig.tight_layout(rect=[0, 0, 1, 1])
        return _fig_to_pil(fig)

    img_noadh = _make_panel(no_adh, "No adhesion", "#9467bd")
    img_adh   = _make_panel(adh,   "adhesion",    "#2ca02c")

    _paste_pil(slide, img_noadh, 0.3, 1.0, 6.3, 5.8)
    _paste_pil(slide, img_adh,   6.8, 1.0, 6.3, 5.8)


def _slide_setup(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _header(slide, "Experiment Setup — 4 Configurations",
            subtitle="All share: B2 DS1 data · same 5-fold splits · "
                     "500 epochs · batch=128 · SupCon loss · coord-based online crop")

    configs = [
        ("ps=32  lat=12  proj=8",  32, 12,  8, "classic baseline",     "#aaaaaa"),
        ("ps=32  lat=64  proj=32", 32, 64, 32, "isolates patch size",   "#1f77b4"),
        ("ps=64  lat=12  proj=8",  64, 12,  8, "isolates latent dim",   "#ff7f0e"),
        ("ps=64  lat=64  proj=32", 64, 64, 32, "this run ✓ done",       "#2ca02c"),
    ]

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.axis("off")
    headers  = ["Configuration", "Patch size", "Latent dim", "Proj dim",
                "Model params", "Notes", "Status"]
    from subcellae.modelling.autoencoders import ContrastiveAE
    rows_data = []
    for label, ps, lat, proj, note, _ in configs:
        model = ContrastiveAE(latent_dim=lat, proj_dim=proj, input_ps=ps,
                              no_ch=1, noise_prob=0.0, BN_flag=False, output_sigmoid=False)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        status = "✓ done" if ps == 64 and lat == 64 else "queued"
        rows_data.append([label, f"{ps}×{ps}", lat, proj, f"{n_params:.2f}M", note, status])
    tbl = ax.table(cellText=rows_data, colLabels=headers,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 2.2)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cfg_color = configs[r - 1][5]
            if c == 0:
                cell.set_facecolor(cfg_color + "33")
            elif r % 2 == 0:
                cell.set_facecolor("#f8f8f8")
    ax.set_title("2×2 grid: patch size × latent dim", fontsize=11,
                 fontweight="bold", pad=10)
    fig.tight_layout()
    _paste_pil(slide, _fig_to_pil(fig), 0.5, 1.0, 12.3, 4.5)

    _txt(slide,
         "Training: SupCon loss (λ_recon=1.0  λ_contrast=0.5  λ_supcon=5.0)  ·  "
         "lr=0.001  weight_decay=1e-4  ·  val_split=0.2 (internal)  ·  "
         "group_split=true  ·  intensity_scale_range=[0.8, 1.2]",
         0.5, 5.7, SLIDE_W - 1.0, 0.7, size=9, color="#555555")


def _slide_cls_results(prs, results: dict):
    """results: {run_tag: {'accs': [f0..f4], 'mean': x, 'std': x}} or None if not done."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _header(slide, "Classification Results — 5-fold CV Balanced Accuracy (No adh vs adh)",
            subtitle="LightGBM on latent features · train fold latents → test fold latents · "
                     "pending runs shown as placeholders")

    labels_short = {
        "ms_b2_ds1_ps32_lat12p8":  "ps=32\nlat=12",
        "ms_b2_ds1_ps32_lat64p32": "ps=32\nlat=64",
        "ms_b2_ds1_ps64_lat12p8":  "ps=64\nlat=12",
        "ms_b2_ds1_ps64_lat64p32": "ps=64\nlat=64",
    }
    bar_colors = {
        "ms_b2_ds1_ps32_lat12p8":  "#aaaaaa",
        "ms_b2_ds1_ps32_lat64p32": "#1f77b4",
        "ms_b2_ds1_ps64_lat12p8":  "#ff7f0e",
        "ms_b2_ds1_ps64_lat64p32": "#2ca02c",
    }
    order = list(labels_short.keys())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5),
                             gridspec_kw={"width_ratios": [2, 1]})

    # Left: per-fold bar chart
    ax = axes[0]
    fold_x = np.arange(N_FOLDS)
    width  = 0.18
    for i, tag in enumerate(order):
        info = results.get(tag)
        if info is None:
            continue
        accs = info["accs"]
        x    = fold_x + (i - 1.5) * width
        bars = ax.bar(x, accs, width, label=labels_short[tag].replace("\n", " "),
                      color=bar_colors[tag], alpha=0.85, edgecolor="white")
    ax.set_xticks(fold_x)
    ax.set_xticklabels([f"fv{k}" for k in range(N_FOLDS)])
    ax.set_ylim(0.5, 1.02)
    ax.axhline(1.0, color="#cccccc", lw=0.8, ls="--")
    ax.set_ylabel("Balanced accuracy", fontsize=10)
    ax.set_title("Per-fold balanced accuracy", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)

    # Right: mean ± std bar
    ax2 = axes[1]
    done_tags   = [t for t in order if results.get(t) is not None]
    done_means  = [results[t]["mean"] for t in done_tags]
    done_stds   = [results[t]["std"]  for t in done_tags]
    done_colors = [bar_colors[t] for t in done_tags]
    done_labels = [labels_short[t].replace("\n", " ") for t in done_tags]
    ax2.barh(done_labels, done_means, xerr=done_stds,
             color=done_colors, alpha=0.85, edgecolor="white",
             error_kw={"ecolor": "#333333", "capsize": 4})
    ax2.set_xlim(0.5, 1.02)
    ax2.axvline(1.0, color="#cccccc", lw=0.8, ls="--")
    ax2.set_xlabel("Mean balanced accuracy", fontsize=10)
    ax2.set_title("Mean ± std", fontsize=11, fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)
    for i, (m, s) in enumerate(zip(done_means, done_stds)):
        ax2.text(m + s + 0.005, i, f"{m:.4f}", va="center", fontsize=9)

    pending = [labels_short[t].replace("\n", " ")
               for t in order if results.get(t) is None]
    if pending:
        ax2.text(0.52, -0.6, f"Pending: {', '.join(pending)}",
                 fontsize=8, color="#888888", style="italic",
                 transform=ax2.get_yaxis_transform())

    fig.tight_layout()
    _paste_pil(slide, _fig_to_pil(fig), 0.4, 1.0, 12.5, 5.8)


def _slide_reconstruction(prs, examples: pd.DataFrame):
    """Show input → reconstruction for example patches using fv0 model."""
    import torch
    from subcellae.modelling.autoencoders import ContrastiveAE

    run_dir = RUN_ROOT_PS64 / f"{RUN_TAG_PS64}_fv0"
    ckpt    = run_dir / "model_best.pt"
    if not ckpt.exists():
        return

    device = "cpu"
    model  = ContrastiveAE(latent_dim=64, proj_dim=32, input_ps=64,
                           no_ch=1, noise_prob=0.0, BN_flag=False,
                           output_sigmoid=False).to(device)
    saved  = torch.load(str(ckpt), map_location=device, weights_only=False)
    state  = saved.state_dict() if hasattr(saved, "state_dict") else saved
    model.load_state_dict(state)
    model.eval()

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _header(slide, "Reconstruction Examples — ps=64 (fv0 model)",
            subtitle="Left: input patch (ps=64)  ·  Middle: reconstruction  ·  "
                     "Right: |input − recon| error  ·  Top: No adhesion  ·  Bottom: adhesion")

    n_per_class = len(examples) // 2
    no_adh = examples[examples["label"] == "No adhesion"].head(n_per_class)
    adh    = examples[examples["label"] == "adhesion"].head(n_per_class)

    def _recon_row(subset, label, color):
        patches_in, patches_rec = [], []
        for _, row in subset.iterrows():
            raw = _crop_patch(row["condition_name"], int(row["frame_idx"]),
                              int(row["cx"]), int(row["cy"]), 64)
            norm = _normalise_patch(raw)
            x    = torch.tensor(norm[np.newaxis, np.newaxis], dtype=torch.float32)
            with torch.no_grad():
                x_hat, _ = model(x)
            inp_np  = norm
            rec_np  = x_hat[0, 0].cpu().numpy()
            # re-normalise recon for display
            rec_norm = _normalise_patch(rec_np)
            patches_in.append(inp_np)
            patches_rec.append(rec_norm)
        return patches_in, patches_rec

    patches_noadh_in,  patches_noadh_rec  = _recon_row(no_adh, "No adhesion", "#9467bd")
    patches_adh_in,    patches_adh_rec    = _recon_row(adh,    "adhesion",    "#2ca02c")

    n = n_per_class
    fig, axes = plt.subplots(4, n, figsize=(n * 1.5, 6.5))
    if n == 1:
        axes = axes[:, np.newaxis]

    row_labels = [
        ("No adh – input",  "#9467bd"), ("No adh – recon",  "#9467bd"),
        ("adh – input",     "#2ca02c"), ("adh – recon",     "#2ca02c"),
    ]
    all_rows = [patches_noadh_in, patches_noadh_rec, patches_adh_in, patches_adh_rec]

    for r_idx, (patches, (rl, rc)) in enumerate(zip(all_rows, row_labels)):
        for col, patch in enumerate(patches):
            axes[r_idx, col].imshow(patch, cmap="gray", vmin=0, vmax=1)
            axes[r_idx, col].axis("off")
        axes[r_idx, 0].set_ylabel(rl, fontsize=8, color=rc, labelpad=2)

    fig.suptitle("Top: No adhesion  ·  Bottom: adhesion  "
                 "  |  Odd rows: input, Even rows: reconstruction",
                 fontsize=9, color="#555555")
    fig.tight_layout()
    _paste_pil(slide, _fig_to_pil(fig), 0.3, 1.0, 12.7, 6.3)


FA5_COLORS_UMAP = {
    "No adhesion":        "#9467bd",
    "Nascent Adhesion":   "#1f77b4",
    "focal complex":      "#ff7f0e",
    "focal adhesion":     "#2ca02c",
    "fibrillar adhesion": "#d62728",
    "Uncertain":          "#aaaaaa",
}
FA5_SHORT_UMAP = {
    "No adhesion":        "No Adh",
    "Nascent Adhesion":   "NA",
    "focal complex":      "FC",
    "focal adhesion":     "FA",
    "fibrillar adhesion": "Fib",
    "Uncertain":          "Unc",
}
FA5_ORDER_UMAP = ["No adhesion", "Nascent Adhesion", "focal complex",
                  "focal adhesion", "fibrillar adhesion", "Uncertain"]


def _collect_umap_data():
    """Load training-fold latents from all folds. Returns (Z, labels_2cls, labels_5cls, cond)."""
    try:
        import umap as umap_lib
    except ImportError:
        try:
            import umap.umap_ as umap_lib
        except ImportError:
            print("  WARNING: umap not available — skipping UMAP slides")
            return None

    fs = pd.read_csv(LAB_DIR / "fold_splits.csv")

    def coord_key(row):
        return f"{row['condition_name']}_f{int(row['frame_idx']):04d}_cx{int(row['cx'])}_cy{int(row['cy'])}"

    all_z, all_labels, all_labels_5cls, all_cond = [], [], [], []
    for k in range(N_FOLDS):
        lat_path = RUN_ROOT_PS64 / f"{RUN_TAG_PS64}_fv{k}" / "latents.csv"
        if not lat_path.exists():
            continue
        lat    = pd.read_csv(lat_path).set_index("filename")
        z_cols = [c for c in lat.columns if c.startswith("z_") and "_proj" not in c]
        for _, row in fs[fs["fold"] != k].iterrows():
            key = coord_key(row)
            if key in lat.index:
                all_z.append(lat.loc[key, z_cols].values.astype(float))
                all_labels.append(row["label"])
                all_labels_5cls.append(row["label_5cls"])
                all_cond.append(row["condition_name"])

    if not all_z:
        return None

    Z = np.array(all_z)
    if len(Z) > 3000:
        idx = np.random.default_rng(42).choice(len(Z), 3000, replace=False)
        Z               = Z[idx]
        all_labels      = [all_labels[i]      for i in idx]
        all_labels_5cls = [all_labels_5cls[i] for i in idx]
        all_cond        = [all_cond[i]        for i in idx]

    print(f"  Running UMAP on {len(Z)} latents (dim={Z.shape[1]})...")
    emb = umap_lib.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                        random_state=42, verbose=False).fit_transform(Z)

    return emb, Z, all_labels, all_labels_5cls, all_cond


def _slide_umap(prs, umap_data):
    if umap_data is None:
        return
    emb, Z, all_labels, all_labels_5cls, all_cond = umap_data
    n = len(emb)

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _header(slide, "UMAP — Latent Space (ps=64, lat=64)  [training fold patches]",
            subtitle=f"{n} patches shown  ·  "
                     "uses training-fold latents (test-fold will be added when eval script runs)")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0))
    for ax, color_by, color_map, title in [
        (axes[0], all_labels, LABEL_COLORS,   "2-class: No adhesion / adhesion"),
        (axes[1], all_cond,   COND_COLORS,    "Condition: ctrl / ycomp"),
    ]:
        for val in sorted(set(color_by)):
            mask  = np.array([v == val for v in color_by])
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=color_map.get(val, "#999999"), s=3,
                       alpha=0.5, label=val, rasterized=True)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(markerscale=4, fontsize=9, framealpha=0.7)
        ax.set_xlabel("UMAP 1", fontsize=9)
        ax.set_ylabel("UMAP 2", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("UMAP of training-fold latents  (ps=64  lat=64  proj=32)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    _paste_pil(slide, _fig_to_pil(fig), 0.4, 1.0, 12.5, 6.0)


def _slide_umap_5cls(prs, umap_data):
    """UMAP coloured by 5-class FA labels."""
    if umap_data is None:
        return
    emb, Z, all_labels, all_labels_5cls, all_cond = umap_data
    n = len(emb)

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _header(slide, "UMAP — Latent Space Coloured by 5-class FA Label",
            subtitle=f"{n} patches shown  ·  ps=64  lat=64  proj=32  ·  training-fold latents")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for cls in FA5_ORDER_UMAP:
        mask = np.array([v == cls for v in all_labels_5cls])
        if mask.sum() == 0:
            continue
        color = FA5_COLORS_UMAP.get(cls, "#aaaaaa")
        short = FA5_SHORT_UMAP.get(cls, cls)
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=color, s=4, alpha=0.6,
                   label=f"{short}  (n={mask.sum()})", rasterized=True)

    ax.set_title("5-class FA label", fontsize=12, fontweight="bold")
    ax.legend(markerscale=3, fontsize=10, framealpha=0.8,
              loc="best", title="FA class")
    ax.set_xlabel("UMAP 1", fontsize=10)
    ax.set_ylabel("UMAP 2", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _paste_pil(slide, _fig_to_pil(fig), 1.5, 0.9, 10.3, 6.4)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    fs = pd.read_csv(LAB_DIR / "fold_splits.csv")
    examples = _pick_examples(fs, n_per_class=6)

    prs = Presentation()
    prs.slide_width  = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)

    print("Slide 1: dataset overview...")
    _slide_dataset(prs, fs)

    print("Slide 2: example patches ps=32 vs ps=64...")
    _slide_example_patches(prs, examples)

    print("Slide 3: setup comparison...")
    _slide_setup(prs)

    print("Slide 4: classification results...")
    results = {
        "ms_b2_ds1_ps64_lat64p32": {
            "accs": [0.9483, 0.9786, 0.9675, 0.9601, 0.9524],
            "mean": 0.9614,
            "std":  0.0109,
        },
        "ms_b2_ds1_ps32_lat12p8":  None,
        "ms_b2_ds1_ps32_lat64p32": None,
        "ms_b2_ds1_ps64_lat12p8":  None,
    }
    _slide_cls_results(prs, results)

    print("Slide 5: reconstruction examples...")
    _slide_reconstruction(prs, examples)

    print("Slide 6+7: UMAP...")
    umap_data = _collect_umap_data()
    _slide_umap(prs, umap_data)
    _slide_umap_5cls(prs, umap_data)

    (REPO / "results").mkdir(exist_ok=True)
    prs.save(str(OUT))
    print(f"\nSaved: {OUT}  ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
