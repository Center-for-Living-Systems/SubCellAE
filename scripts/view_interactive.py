"""
Interactive Patch Viewer — Panel + Bokeh

Two-direction exploration:

  Direction A — UMAP → detail
    Hover a UMAP dot  →  raw + recon patch appear in tooltip (instant, client-side).
    Tap a UMAP dot    →  detail panel below updates (patch images + prediction text).

  Direction B — Image → UMAP
    Choose an image from the dropdown (condition × source image).
    The full paxillin canvas is shown with coloured patch rectangles.
    Click anywhere on the canvas  →  the nearest patch is found, its UMAP point is
    highlighted with a large red dot, and the prediction text updates below.

Layout
------
  ┌──────────────────────┬────────────────────────────┐
  │  [Color ▼]           │  [Image selector ▼]        │
  │  UMAP scatter        │  Full paxillin canvas       │
  │  (hover = tooltip)   │  (click = UMAP highlight)   │
  │  (tap   = detail ↓)  │  (colored patch boxes)      │
  ├──────────────────────┴────────────────────────────┤
  │  FA: …  Position: …   |  Raw patch  |  Recon patch │
  └─────────────────────────────────────────────────────┘

Usage
-----
    python scripts/view_interactive.py path/to/interactive.h5
    panel serve scripts/view_interactive.py --args path/to/interactive.h5 --show
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import h5py
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import panel as pn
from bokeh.events import Tap
from bokeh.layouts import column as bk_column
from bokeh.models import (
    ColumnDataSource, CustomJS, HoverTool,
    LinearColorMapper, Range1d, Select,
)
from bokeh.plotting import figure

pn.extension(sizing_mode='stretch_width')

# ── Shared colour palettes (authoritative source: subcellae.utils.label_colors) ──
from subcellae.utils.label_colors import (
    classification_label_order as FA_ORDER,
    classification_label_to_color as FA_COLOR_MAP,
    position_label_order as POS_ORDER,
    position_label_to_color as POS_COLOR_MAP,
)
FALLBACK = "#cccccc"

# Grayscale palette: index 0 → black, index 255 → white
try:
    from bokeh.palettes import gray as _bk_gray
    GRAY256 = _bk_gray(256)
except Exception:
    GRAY256 = [f'#{i:02x}{i:02x}{i:02x}' for i in range(256)]


def _label_color(label: str, color_map: dict) -> str:
    return color_map.get(str(label), FALLBACK)


# ── HDF5 loading ──────────────────────────────────────────────────────────────

def _read_h5_model(path: str):
    """Load model-side data from an interactive.h5 / model.h5.

    Returns (df, patches_raw, patches_recon, images_raw, img_meta,
             pad, scale, result_dir, plots).
    """
    with h5py.File(path, 'r') as f:
        df            = pd.read_csv(io.StringIO(f['meta/csv'][()].decode()))
        patches_raw   = f['patches/raw'][()]   if 'patches/raw'   in f else None
        patches_recon = f['patches/recon'][()] if 'patches/recon' in f else None
        images_raw    = f['images/raw'][()]    if 'images/raw'    in f else None
        img_meta      = (pd.read_csv(io.StringIO(f['images/meta'][()].decode()))
                         if 'images/meta' in f else None)
        pad_size    = float(f.attrs.get('pad_size', 64))
        image_scale = float(f.attrs.get('image_scale', 1.0))
        result_dir  = Path(str(f.attrs.get('result_dir', '')))
        plots = {key: bytes(f[f'plots/{key}'][()])
                 for key in f.get('plots', {}).keys()}
    return df, patches_raw, patches_recon, images_raw, img_meta, pad_size, image_scale, result_dir, plots


def _read_h5_data(path: str):
    """Load dataset-side data from a data.h5 (patches + images, no model outputs).

    Returns (df_data, patches_raw, images_raw, img_meta, pad, scale).
    """
    with h5py.File(path, 'r') as f:
        df_data     = pd.read_csv(io.StringIO(f['meta/csv'][()].decode()))
        patches_raw = f['patches/raw'][()] if 'patches/raw' in f else None
        images_raw  = f['images/raw'][()]  if 'images/raw'  in f else None
        img_meta    = (pd.read_csv(io.StringIO(f['images/meta'][()].decode()))
                       if 'images/meta' in f else None)
        pad_size    = float(f.attrs.get('pad_size', 64))
        image_scale = float(f.attrs.get('image_scale', 1.0))
    return df_data, patches_raw, images_raw, img_meta, pad_size, image_scale


def load_sources(data_h5: str | None, model_h5: str | None):
    """Load from data.h5 + model H5, or from a single legacy interactive.h5.

    Two-file mode  (data_h5 AND model_h5 provided):
      • images + raw patches come from data_h5
      • latents / UMAP / predictions / recon come from model_h5
      • static columns (mean_intensity, annotation_label) merged from data_h5

    Single-file mode (only model_h5 provided):
      • backward-compatible: reads everything from one interactive.h5

    Returns the same 9-tuple as the old load_h5().
    """
    if data_h5 and model_h5:
        print(f'[view] data  : {data_h5}')
        print(f'[view] model : {model_h5}')
        df_model, pr_model, patches_recon, im_model, imm_model, pad_m, scale_m, result_dir, plots = \
            _read_h5_model(model_h5)
        df_data, pr_data, im_data, imm_data, pad_d, scale_d = \
            _read_h5_data(data_h5)

        # Prefer data.h5 for images (higher quality / all conditions)
        images_raw = im_data   if im_data   is not None else im_model
        img_meta   = imm_data  if imm_data  is not None else imm_model
        pad_size   = pad_d
        image_scale = scale_d

        # Merge static columns from data.h5 into model df (join on filename)
        static_cols = [c for c in ('filename', 'mean_intensity',
                                   'annotation_label', 'annotation_label_name',
                                   'canvas_cx', 'canvas_cy', 'ps')
                       if c in df_data.columns]
        # drop cols already in df_model to avoid duplicate suffixes
        drop_from_data = [c for c in static_cols if c != 'filename' and c in df_model.columns]
        df = df_model.merge(df_data[static_cols].drop(columns=drop_from_data),
                            on='filename', how='left')

        # Reindex patches_raw from data.h5 to match model df row order
        if pr_data is not None:
            fname_to_idx = {str(fn): i for i, fn in enumerate(df_data['filename'])}
            h, w = pr_data.shape[1], pr_data.shape[2]
            patches_raw = np.zeros((len(df), h, w), dtype=pr_data.dtype)
            for i, fn in enumerate(df['filename']):
                j = fname_to_idx.get(str(fn))
                if j is not None:
                    patches_raw[i] = pr_data[j]
        else:
            patches_raw = pr_model

        return df, patches_raw, patches_recon, images_raw, img_meta, pad_size, image_scale, result_dir, plots

    elif model_h5:
        print(f'[view] Loading (single-file) {model_h5}')
        return _read_h5_model(model_h5)

    elif data_h5:
        print(f'[view] Loading (data-only) {data_h5}')
        df_data, patches_raw, images_raw, img_meta, pad_size, image_scale = _read_h5_data(data_h5)
        return df_data, patches_raw, None, images_raw, img_meta, pad_size, image_scale, Path(''), {}

    else:
        raise ValueError('At least one H5 path is required.')


# kept for backward compat (external callers)
def load_h5(path: str):
    return load_sources(None, path)


# ── Image helpers ─────────────────────────────────────────────────────────────

def _norm_image(arr: np.ndarray) -> np.ndarray:
    """Cast to float32 for display; data is pre-normalized to [0, 1]."""
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    return arr.astype(np.float32)


def _make_pixel_overlay(arr: np.ndarray, patch_mask: np.ndarray,
                        show_blue: bool, show_yellow: bool, show_red: bool) -> np.ndarray:
    """Return a (H, W) uint32 RGBA array for Bokeh image_rgba (y-flipped).

    Only pixels where patch_mask is True are considered.
    Encoding: R | G<<8 | B<<16 | A<<24  (little-endian, α=160).
      Blue   #4488FF : pixels < 0
      Yellow #FFCC00 : pixels in (2, 5]
      Red    #FF4444 : pixels > 5
    """
    H, W = arr.shape[:2]
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    if show_blue:
        rgba[patch_mask & (arr < 0)] = [0x44, 0x88, 0xFF, 160]
    if show_yellow:
        rgba[patch_mask & (arr > 2) & (arr <= 5)] = [0xFF, 0xCC, 0x00, 160]
    if show_red:
        rgba[patch_mask & (arr > 5)] = [0xFF, 0x44, 0x44, 160]
    packed = np.ascontiguousarray(rgba).view(np.uint32).reshape(H, W)
    return np.ascontiguousarray(np.flipud(packed))


def _patch_intensity_overlay(arr: np.ndarray, show_blue: bool,
                              show_yellow: bool, show_red: bool) -> np.ndarray | None:
    """Return float32 RGBA (H, W, 4) overlay for matplotlib imshow, or None."""
    if not (show_blue or show_yellow or show_red):
        return None
    ov = np.zeros((*arr.shape[:2], 4), dtype=np.float32)
    if show_blue:
        ov[arr < 0]                   = [0x44/255, 0x88/255, 1.0,      160/255]
    if show_yellow:
        ov[(arr > 2) & (arr <= 5)]    = [1.0,      0xCC/255, 0.0,      160/255]
    if show_red:
        ov[arr > 5]                   = [1.0,      0x44/255, 0x44/255, 160/255]
    return ov


def _flip_for_bokeh(arr: np.ndarray) -> np.ndarray:
    """Flip vertically so array row-0 renders at the top of a Bokeh figure.

    Bokeh's 'image' renderer places array row-0 at y=0 (bottom).  Flipping
    gives the correct top-down orientation for microscopy images.

    Coordinate mapping after flipud:
        Bokeh y-coordinate  ↔  original array row (H - 1 - bokeh_y ≈ array_row)
        For a tap at bokeh_y:  array_row ≈ H - bokeh_y
    """
    return np.ascontiguousarray(np.flipud(arr))


def _get_frame(images_raw: np.ndarray, img_meta: pd.DataFrame,
               group: str, channel: int = 0) -> np.ndarray | None:
    """Return the canvas array for a given (group, channel)."""
    matches = img_meta[img_meta['group'].astype(str) == group]
    if matches.empty:
        return None
    if 'channel' in matches.columns:
        ch0 = matches[matches['channel'] == channel]
        row = ch0.iloc[0] if not ch0.empty else matches.iloc[0]
    else:
        row = matches.iloc[0]
    return images_raw[int(row['frame'])]


# ── Matplotlib figure helpers ─────────────────────────────────────────────────

def _fig_to_pane(fig: plt.Figure, dpi: int = 130) -> pn.pane.PNG:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return pn.pane.PNG(buf, sizing_mode='scale_width')


def _patch_figure(raw: np.ndarray, recon: np.ndarray | None = None,
                  title: str = '',
                  show_blue: bool = False, show_yellow: bool = False,
                  show_red: bool = False) -> pn.pane.PNG:
    panels = [('Raw', raw)] + ([('Recon', recon)] if recon is not None else [])
    fig, axes = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 2.6))
    if len(panels) == 1:
        axes = [axes]
    for ax, (lbl, arr) in zip(axes, panels):
        ax.imshow(arr, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        ov = _patch_intensity_overlay(arr, show_blue, show_yellow, show_red)
        if ov is not None:
            ax.imshow(ov, interpolation='nearest')
        ax.set_title(lbl, fontsize=10)
        ax.axis('off')
    if title:
        fig.suptitle(title, fontsize=8, y=1.01)
    fig.tight_layout(pad=0.3)
    return _fig_to_pane(fig)


# ── Legend HTML ───────────────────────────────────────────────────────────────

def _legend_html() -> str:
    rows = ['<div style="font-size:11px;line-height:1.8;"><b>FA type</b>']
    for lbl in FA_ORDER:
        col = FA_COLOR_MAP.get(lbl, FALLBACK)
        rows.append(
            f'<span style="background:{col};display:inline-block;'
            f'width:11px;height:11px;margin-right:5px;border-radius:2px;'
            f'vertical-align:middle;"></span>{lbl}'
        )
    rows.append('<br><b>Position</b>')
    for lbl in POS_ORDER:
        col = POS_COLOR_MAP.get(lbl, FALLBACK)
        rows.append(
            f'<span style="background:{col};display:inline-block;'
            f'width:11px;height:11px;margin-right:5px;border-radius:2px;'
            f'vertical-align:middle;"></span>{lbl}'
        )
    rows.append('</div>')
    return '<br>'.join(rows)


# ── Main app ──────────────────────────────────────────────────────────────────

def build_app(data_h5: str | None = None,
              model_h5: str | None = None) -> pn.viewable.Viewable:
    (df, patches_raw, patches_recon,
     images_raw, img_meta, pad_size, image_scale, result_dir, plots) = \
        load_sources(data_h5, model_h5)
    n = len(df)
    print(f'[view]   {n} patches, image_scale={image_scale}')


    # ── Old-format fallback: individual TIFF files on disk ────────────────────
    recon_patches_dir = result_dir / 'recon' / 'patches'
    recon_images_dir  = result_dir / 'recon' / 'images'
    has_old_patches = (patches_raw is None and result_dir != Path('')
                       and recon_patches_dir.is_dir())
    old_img_files: list = []
    has_old_images = False
    if images_raw is None and img_meta is None and result_dir != Path(''):
        old_img_files = sorted(recon_images_dir.glob('raw_*.tif'))
        has_old_images = len(old_img_files) > 0
    if has_old_patches:
        print(f'[view]   Old-format patches found in {recon_patches_dir}')
    if has_old_images:
        print(f'[view]   Old-format images: {len(old_img_files)} files')

    # Derive model/variant name from result_dir or model_h5 path
    if result_dir != Path('') and result_dir.parent.name:
        model_name = result_dir.parent.name
    elif model_h5:
        model_name = Path(model_h5).stem
    else:
        model_name = 'data-only'
    print(f'[view]   Model: {model_name}')

    # Disk fallback for MSE plots not packed into H5
    _plot_names = ['mse_distribution', 'mse_by_condition_split']
    if result_dir != Path(''):
        for pname in _plot_names:
            if pname not in plots:
                p = result_dir / 'analysis' / f'{pname}.png'
                if p.exists():
                    plots[pname] = p.read_bytes()
                    print(f'[view]   Loaded plot from disk: {pname}.png')

    # Detect data-only mode: no latents and no UMAP
    z_cols   = [c for c in df.columns if c.startswith('z_')]
    has_umap = 'UMAP_1' in df.columns or 'UMAP_2' in df.columns
    data_only = not has_umap and not z_cols
    print(f'[view]   data_only={data_only}')

    # Fall back to latent dims if UMAP not available (model mode only)
    if not data_only:
        if not has_umap:
            df['UMAP_1'], df['UMAP_2'] = df[z_cols[0]], df[z_cols[1]]
            print(f'[view]   No UMAP -- showing {z_cols[0]} vs {z_cols[1]}')

    fa_pred  = df.get('fa_pred',  pd.Series([''] * n)).fillna('').astype(str)
    pos_pred = df.get('pos_pred', pd.Series([''] * n)).fillna('').astype(str)

    # ── Data-only left panel: patch max histogram ─────────────────────────────
    if data_only:
        cond_col = 'condition_name' if 'condition_name' in df.columns else 'condition'
        conds    = df[cond_col].fillna('').unique() if cond_col in df.columns else []

        px_vals = (patches_raw.flatten().astype(np.float32)
                   if patches_raw is not None else np.zeros(0, dtype=np.float32))
        fig_hist, ax = plt.subplots(figsize=(5.2, 5.0))
        ax.hist(px_vals, bins=200, color='#888888', alpha=0.75, density=True)
        ax.axvline(0, color='#4488FF', lw=1.6, ls='--', label='< 0')
        ax.axvline(2, color='#FFCC00', lw=1.6, ls='--', label='> 2')
        ax.axvline(5, color='#FF4444', lw=1.6, ls='--', label='> 5')
        ax.set_xlabel('pixel intensity')
        ax.set_ylabel('density')
        n_blue   = int((px_vals < 0).sum())
        n_yellow = int((px_vals > 2).sum())
        n_red    = int((px_vals > 5).sum())
        ax.set_title(
            f'All patches  N={n}  ({len(px_vals):,} px)\n'
            f'<0: {n_blue}  >2: {n_yellow}  >5: {n_red}',
            fontsize=10,
        )
        ax.legend(fontsize=9)
        fig_hist.tight_layout()
        left_col = pn.Column(
            pn.pane.HTML(
                f'<div style="font-size:11px;line-height:1.9;">'
                f'<b>Dataset:</b> {data_h5 or ""}<br>'
                f'<b>Patches:</b> {n}<br>'
                f'<b>Conditions:</b> {", ".join(str(c) for c in conds)}</div>',
                width=480,
            ),
            _fig_to_pane(fig_hist, dpi=120),
            width=490,
        )
        highlight_src = ColumnDataSource({'x': [], 'y': []})  # unused but ref'd later
    else:
        # ── UMAP ColumnDataSource ─────────────────────────────────────────────
        umap_data: dict = dict(
            x         = df['UMAP_1'].fillna(0).values,
            y         = df['UMAP_2'].fillna(0).values,
            idx       = np.arange(n, dtype=int),
            condition = (df.get('condition_name', df.get('condition', pd.Series([''] * n)))
                         .fillna('').astype(str).values),
            fa_pred   = fa_pred.values,
            pos_pred  = pos_pred.values,
            filename  = df['filename'].astype(str).values,
            color_fa  = [_label_color(v, FA_COLOR_MAP)  for v in fa_pred],
            color_pos = [_label_color(v, POS_COLOR_MAP) for v in pos_pred],
        )
        umap_data['color'] = list(umap_data['color_fa'])
        has_b64 = 'raw_b64' in df.columns
        if has_b64:
            umap_data['raw_b64']   = df['raw_b64'].values
            umap_data['recon_b64'] = df['recon_b64'].values

        umap_src = ColumnDataSource(umap_data)

        # Single big red dot on UMAP -- updated when user clicks the image panel
        highlight_src = ColumnDataSource({'x': [], 'y': []})

        # ── UMAP scatter figure ───────────────────────────────────────────────
        p_umap = figure(
            width=520, height=500,
            title='UMAP  (hover = patch tooltip  |  tap = detail panel)',
            tools='pan,wheel_zoom,box_zoom,reset,tap',
            toolbar_location='above',
        )

        scatter = p_umap.scatter(
            'x', 'y', source=umap_src, marker='circle',
            fill_color='color', line_color='color',
            size=5, alpha=0.65,
            nonselection_fill_color='color', nonselection_fill_alpha=0.15,
            nonselection_line_alpha=0.0,
            selection_fill_color='color', selection_line_color='white',
            selection_line_width=1.5,
        )

        # Highlighted point (from image click) -- drawn on top as a large red dot
        p_umap.scatter(
            'x', 'y', source=highlight_src, marker='circle',
            fill_color='red', line_color='white',
            size=18, alpha=1.0, line_width=2.5,
        )

        # Hover tooltip with embedded patch images (client-side, instant)
        if has_b64:
            hover_html = """
                <div style="background:#111;padding:6px 8px;border-radius:6px;max-width:310px;">
                  <div style="display:flex;gap:6px;">
                    <div style="text-align:center;">
                      <img src="data:image/png;base64,@raw_b64"
                           style="width:128px;height:128px;image-rendering:pixelated;display:block;"/>
                      <span style="color:#aaa;font-size:10px;">Raw</span>
                    </div>
                    <div style="text-align:center;">
                      <img src="data:image/png;base64,@recon_b64"
                           style="width:128px;height:128px;image-rendering:pixelated;display:block;"/>
                      <span style="color:#aaa;font-size:10px;">Recon</span>
                    </div>
                  </div>
                  <div style="color:#ccc;font-size:10px;margin-top:5px;line-height:1.4;">
                    @filename<br>cond: @condition<br>FA: @fa_pred<br>Pos: @pos_pred
                  </div>
                </div>"""
            p_umap.add_tools(HoverTool(renderers=[scatter], tooltips=hover_html))
        else:
            p_umap.add_tools(HoverTool(renderers=[scatter], tooltips=[
                ('file', '@filename'), ('cond', '@condition'),
                ('FA',   '@fa_pred'),  ('Pos',  '@pos_pred'),
            ]))

        p_umap.xaxis.axis_label = 'UMAP 1'
        p_umap.yaxis.axis_label = 'UMAP 2'

        # Colour-by selector (pure JS -- no server round-trip)
        color_select = Select(
            title='Colour by', value='fa_pred',
            options=[('fa_pred', 'FA type'), ('pos_pred', 'Position')],
            width=180,
        )
        color_select.js_on_change('value', CustomJS(
            args=dict(src=umap_src, plot=p_umap), code="""
            const d = src.data;
            d['color'] = (cb_obj.value === 'fa_pred')
                ? [...d['color_fa']] : [...d['color_pos']];
            src.change.emit();
            const lbl = (cb_obj.value === 'fa_pred') ? 'FA type' : 'Position';
            plot.title.text = 'UMAP  -- ' + lbl
                + '  (hover = patch tooltip  |  tap = detail panel)';
        """))

        left_col = pn.pane.Bokeh(bk_column(color_select, p_umap))
    # end if/else data_only

    # ── Outlier highlight checkboxes (shared with canvas) ─────────────────────
    ck_blue   = pn.widgets.Checkbox(name="< 0  (blue)",   value=False)
    ck_yellow = pn.widgets.Checkbox(name="> 2  (yellow)", value=False)
    ck_red    = pn.widgets.Checkbox(name="> 5  (red)",    value=False)

    # ── Full image Bokeh figure (Direction B) ─────────────────────────────────
    has_images = (images_raw is not None and img_meta is not None) or has_old_images

    # Placeholders updated inside the has_images block
    rects_src = sel_src = img_fig = img_pane = img_select_widget = None
    _state: dict = {}

    if has_images:
        # Build selector options: "condition | group_key"
        pg_col   = 'patch_group' if 'patch_group' in df.columns else 'group'
        cond_col = 'condition_name' if 'condition_name' in df.columns else 'condition'
        grp_to_cond: dict = {}
        for _, row in df[[pg_col, cond_col]].dropna().drop_duplicates().iterrows():
            grp_to_cond[str(row[pg_col])] = str(row[cond_col])

        if images_raw is not None and img_meta is not None:
            # Packed format: images stored in HDF5 array
            unique_groups = sorted(img_meta['group'].astype(str).unique())
            def _get_canvas(group_key: str) -> np.ndarray:
                return _norm_image(_get_frame(images_raw, img_meta, group_key))
        else:
            # Old-format: individual TIFFs on disk (raw_{group_key}.tif)
            unique_groups = sorted(p.stem[4:] for p in old_img_files)  # strip 'raw_'
            def _get_canvas(group_key: str) -> np.ndarray:
                p = recon_images_dir / f'raw_{group_key}.tif'
                arr = tifffile.imread(str(p)).astype(np.float32)
                if arr.ndim == 3:
                    arr = arr[0]
                mx = arr.max()
                return arr / mx if mx > 0 else arr

        img_options = {f"{grp_to_cond.get(g, '?')} | {g}": g
                       for g in unique_groups}

        init_group = unique_groups[0]
        init_arr   = _get_canvas(init_group)
        H, W       = init_arr.shape[:2]

        # Image data source
        img_src = ColumnDataSource(dict(
            image=[_flip_for_bokeh(init_arr)],
            x=[0], y=[0], dw=[W], dh=[H],
        ))

        # Patch rectangle source (coloured by FA prediction)
        rects_src = ColumnDataSource(dict(
            x=[], y=[], width=[], height=[], color=[], df_idx=[],
        ))
        # Selected-patch white-border highlight
        sel_src = ColumnDataSource(dict(x=[], y=[], width=[], height=[]))

        # Figure
        img_fig = figure(
            width=520, height=520,
            x_range=Range1d(0, W),
            y_range=Range1d(0, H),
            title='Full paxillin canvas  (click a patch to highlight on UMAP)',
            tools='tap,pan,wheel_zoom,reset',
            toolbar_location='above',
        )
        gray_mapper = LinearColorMapper(palette=GRAY256, low=0.0, high=1.0)
        img_fig.image(
            image='image', source=img_src,
            x=0, y=0, dw=W, dh=H,
            color_mapper=gray_mapper,
        )
        img_fig.rect(
            'x', 'y', 'width', 'height', source=rects_src,
            fill_alpha=0, line_color='color', line_width=0.9, line_alpha=0.75,
        )
        img_fig.rect(
            'x', 'y', 'width', 'height', source=sel_src,
            fill_alpha=0, line_color='white', line_width=2.5,
        )
        img_fig.xaxis.axis_label = 'column (px)'
        img_fig.yaxis.axis_label = 'row (px)'

        # Pixel-level intensity overlay (RGBA, drawn on top of canvas image)
        overlay_src = ColumnDataSource(dict(
            image=[np.zeros((H, W), dtype=np.uint32)],
            x=[0], y=[0], dw=[W], dh=[H],
        ))
        img_fig.image_rgba(
            image='image', source=overlay_src,
            x='x', y='y', dw='dw', dh='dh',
        )

        def _refresh_overlay() -> None:
            sb, sy, sr = ck_blue.value, ck_yellow.value, ck_red.value
            Hc, Wc = _state['H'], _state['W']
            if not (sb or sy or sr):
                overlay_src.data = dict(
                    image=[np.zeros((Hc, Wc), dtype=np.uint32)],
                    x=[0], y=[0], dw=[Wc], dh=[Hc],
                )
                return
            arr = _get_canvas(_state['group'])
            H, W = arr.shape[:2]
            # Build mask: True only within each patch's bounding box
            mask = np.zeros((H, W), dtype=bool)
            sub = df[df[pg_col].astype(str) == _state['group']]
            for _, row in sub.iterrows():
                cx = row.get('canvas_cx', np.nan)
                cy = row.get('canvas_cy', np.nan)
                ps = int(row.get('ps', 32))
                if pd.isna(cx) or pd.isna(cy):
                    continue
                half = ps // 2
                r0, r1 = max(0, int(cy) - half), min(H, int(cy) + half)
                c0, c1 = max(0, int(cx) - half), min(W, int(cx) + half)
                mask[r0:r1, c0:c1] = True
            overlay_src.data = dict(
                image=[_make_pixel_overlay(arr, mask, sb, sy, sr)],
                x=[0], y=[0], dw=[W], dh=[H],
            )

        # Helper: build rect data for a group (in Bokeh flipped-y coordinates)
        def _rects_for_group(group_key: str, img_H: int) -> dict:
            mask = df[pg_col].astype(str) == group_key
            sub  = df[mask]
            xs, ys, ws, hs, cols, idxs = [], [], [], [], [], []
            for i, row in sub.iterrows():
                cx = row.get('canvas_cx', np.nan)
                cy = row.get('canvas_cy', np.nan)
                ps = int(row.get('ps', 32))
                if pd.isna(cx) or pd.isna(cy):
                    continue
                # With flipud display: Bokeh_y = img_H - canvas_cy
                xs.append(float(cx) * image_scale)
                ys.append((img_H - float(cy)) * image_scale)
                ws.append(float(ps) * image_scale)
                hs.append(float(ps) * image_scale)
                cols.append(_label_color(str(row.get('fa_pred', '')), FA_COLOR_MAP))
                idxs.append(i)
            return dict(x=xs, y=ys, width=ws, height=hs, color=cols, df_idx=idxs)

        _state.update(group=init_group, H=H, W=W)
        rects_src.data = _rects_for_group(init_group, H)

        def _load_group(group_key: str) -> None:
            arr     = _get_canvas(group_key)
            Hn, Wn  = arr.shape[:2]
            img_src.data = dict(
                image=[_flip_for_bokeh(arr)],
                x=[0], y=[0], dw=[Wn], dh=[Hn],
            )
            img_fig.x_range.start, img_fig.x_range.end = 0, Wn
            img_fig.y_range.start, img_fig.y_range.end = 0, Hn
            rects_src.data = _rects_for_group(group_key, Hn)
            sel_src.data   = dict(x=[], y=[], width=[], height=[])
            highlight_src.data = dict(x=[], y=[])
            _state.update(group=group_key, H=Hn, W=Wn)
            _refresh_overlay()

        img_select_widget = pn.widgets.Select(
            name='Image', options=img_options,
            value=init_group, width=420,
        )
        img_select_widget.param.watch(lambda e: _load_group(e.new), 'value')
        img_pane = pn.pane.Bokeh(img_fig)

        ck_blue.param.watch(lambda _: _refresh_overlay(), 'value')
        ck_yellow.param.watch(lambda _: _refresh_overlay(), 'value')
        ck_red.param.watch(lambda _: _refresh_overlay(), 'value')

    # ── Shared detail panel (bottom bar) ──────────────────────────────────────
    pred_md   = pn.pane.HTML(
        '<i style="color:#888;">Hover the UMAP for a quick patch preview.  '
        'Tap the UMAP or click a patch in the canvas for full details.</i>',
        width=300,
    )
    patch_col = pn.Column(pn.pane.Markdown(''), width=300)

    def _show_detail(idx: int) -> None:
        row   = df.iloc[idx]
        fa    = str(row.get('fa_pred',  '--'))
        pos   = str(row.get('pos_pred', '--'))
        fname = str(row.get('filename', ''))
        cond  = str(row.get('condition_name', row.get('condition', '')))
        fa_color  = FA_COLOR_MAP.get(fa,  FALLBACK)
        pos_color = POS_COLOR_MAP.get(pos, FALLBACK)
        pred_md.object = (
            f'<div style="font-size:12px;line-height:2;">'
            f'<b>Patch:</b> <code>{Path(fname).stem}</code><br>'
            f'<b>Condition:</b> {cond}<br>'
            f'<i>Prediction by</i> <code>{model_name}</code>:<br>'
            f'<b>FA type:</b> '
            f'<span style="font-size:14px;font-weight:bold;color:{fa_color};">'
            f'{fa}</span><br>'
            f'<b>Position:</b> '
            f'<span style="font-size:14px;font-weight:bold;color:{pos_color};">'
            f'{pos}</span>'
            f'</div>'
        )
        if patches_raw is not None:
            recon_arr = patches_recon[idx] if patches_recon is not None else None
            patch_col.objects = [_patch_figure(
                patches_raw[idx], recon_arr,
                title=Path(fname).stem,
                show_blue=ck_blue.value,
                show_yellow=ck_yellow.value,
                show_red=ck_red.value,
            )]
        elif has_old_patches:
            stem    = Path(fname).stem
            raw_p   = recon_patches_dir / f'raw_{stem}.tif'
            recon_p = recon_patches_dir / f'recon_{stem}.tif'
            if raw_p.exists() and recon_p.exists():
                raw_arr   = tifffile.imread(str(raw_p)).astype(np.float32)
                recon_arr = tifffile.imread(str(recon_p)).astype(np.float32)
                if raw_arr.ndim == 3:
                    raw_arr = raw_arr[0]
                if recon_arr.ndim == 3:
                    recon_arr = recon_arr[0]
                for _a in [raw_arr, recon_arr]:
                    mx = _a.max()
                    if mx > 0:
                        _a /= mx
                patch_col.objects = [_patch_figure(raw_arr, recon_arr, title=stem)]
            else:
                patch_col.objects = [pn.pane.Markdown(f'*Patch files not found for {stem}*')]

    # ── Direction A: UMAP tap → detail + canvas highlight ─────────────────────
    def _on_umap_tap(attr, old, new):
        if not new:
            return
        idx = int(new[0])
        _show_detail(idx)

        if not has_images:
            return
        row    = df.iloc[idx]
        pg_val = str(row.get(pg_col, ''))
        cx     = row.get('canvas_cx', np.nan)
        cy     = row.get('canvas_cy', np.nan)
        ps     = float(row.get('ps', 32))
        H_cur  = _state['H']
        if not pd.isna(cx) and not pd.isna(cy):
            bx = float(cx) * image_scale
            by = (H_cur - float(cy)) * image_scale
            sel_src.data = dict(
                x=[bx], y=[by],
                width=[ps * image_scale], height=[ps * image_scale],
            )
            # Switch image panel to the source image containing this patch
            if pg_val and pg_val != _state['group']:
                _load_group(pg_val)
                img_select_widget.value = pg_val

    if not data_only:
        umap_src.selected.on_change('indices', _on_umap_tap)

    # ── Direction B: image click → UMAP highlight + detail ───────────────────
    if has_images:
        def _on_image_tap(event: Tap) -> None:
            H_cur       = _state['H']
            # Convert Bokeh click coordinates back to canvas (original array) space
            canvas_cx_t = event.x / image_scale
            canvas_cy_t = (H_cur - event.y) / image_scale   # undo flipud

            # Patch centres in canvas coordinates
            xs_bk  = np.array(rects_src.data['x'],      dtype=float)
            ys_bk  = np.array(rects_src.data['y'],      dtype=float)
            df_idx = np.array(rects_src.data['df_idx'], dtype=int)

            if len(xs_bk) == 0:
                return

            # Convert rect centres back to canvas coords for distance calculation
            cx_arr = xs_bk / image_scale
            cy_arr = (H_cur - ys_bk) / image_scale

            dists   = np.sqrt((cx_arr - canvas_cx_t)**2 + (cy_arr - canvas_cy_t)**2)
            near_i  = int(np.argmin(dists))
            near_df = int(df_idx[near_i])

            # Big red dot on UMAP (model mode only)
            if not data_only:
                row    = df.iloc[near_df]
                umap_x = float(row.get('UMAP_1', row.get('umap_1', 0)) or 0)
                umap_y = float(row.get('UMAP_2', row.get('umap_2', 0)) or 0)
                highlight_src.data = dict(x=[umap_x], y=[umap_y])

            # White-border highlight on canvas
            sel_src.data = dict(
                x=[xs_bk[near_i]],
                y=[ys_bk[near_i]],
                width=[rects_src.data['width'][near_i]],
                height=[rects_src.data['height'][near_i]],
            )

            _show_detail(near_df)

        img_fig.on_event(Tap, _on_image_tap)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend = pn.pane.HTML(_legend_html(), width=300)

    # ── MSE plot button ───────────────────────────────────────────────────────
    import base64
    if plots:
        def _b64(key: str) -> str:
            return base64.b64encode(plots.get(key, b'')).decode('ascii')
        mse_button = pn.widgets.Button(
            name='Show MSE plots', button_type='primary', width=200,
        )
        mse_button.js_on_click(
            args=dict(d=_b64('mse_distribution'),
                      c=_b64('mse_by_condition_split')),
            code="""
            var w = window.open('', '_blank', 'width=1300,height=620');
            w.document.write(
                '<html><head><title>MSE plots</title></head>'
                + '<body style="background:#1a1a1a;display:flex;gap:16px;'
                + 'padding:20px;justify-content:center;align-items:flex-start;">'
                + '<img src="data:image/png;base64,' + d
                + '" style="max-height:560px;border-radius:6px;"/>'
                + '<img src="data:image/png;base64,' + c
                + '" style="max-height:560px;border-radius:6px;"/>'
                + '</body></html>');
            w.document.close();
            """,
        )
    else:
        mse_button = pn.pane.Markdown('*MSE plots not found.*', width=200)

    # ── Layout assembly ───────────────────────────────────────────────────────
    detail_col = pn.Column(
        pn.pane.Markdown('### Details', width=300),
        pred_md,
        patch_col,
        pn.layout.Divider(),
        legend,
        pn.layout.Divider(),
        mse_button,
        width=320,
    )

    if has_images:
        outlier_row = pn.Row(
            pn.pane.HTML('<b style="font-size:11px;">Highlight:</b>', width=65),
            ck_blue, ck_yellow, ck_red,
        )
        canvas_col = pn.Column(img_select_widget, outlier_row, img_pane, width=540)
    else:
        canvas_col = pn.pane.Markdown(
            '*Full image data not in this HDF5.*\n\n'
            'Re-pack with `--image-scale 1.0` (default) to include canvas images.',
            width=540,
        )

    src_label = (Path(model_h5).name if model_h5 else '') + \
                (' + ' + Path(data_h5).name if data_h5 else '')
    return pn.Column(
        pn.pane.HTML(
            f'<h2>Interactive Patch Viewer &nbsp;·&nbsp; '
            f'<code>{src_label}</code>'
            f' &nbsp;·&nbsp; <code>{model_name}</code></h2>',
            sizing_mode='stretch_width',
        ),
        pn.Row(left_col, pn.Spacer(width=12), canvas_col,
               pn.Spacer(width=12), detail_col),
    )


def _get_cli_paths() -> tuple[str | None, str | None]:
    """Return (data_h5, model_h5) from CLI/session args.

    Accepted forms:
      --args data.h5 model.h5   → two-file mode
      --args model.h5           → single-file legacy mode
      (no args)                 → show loader UI
    """
    def _decode(a):
        return a.decode() if isinstance(a, bytes) else str(a)

    sess  = pn.state.session_args
    raw   = sess.get('args', []) or []
    parts = [_decode(a) for a in raw] if raw else sys.argv[1:]

    if len(parts) >= 2:
        return parts[0], parts[1]
    if len(parts) == 1:
        return None, parts[0]
    return None, None


# ── Entry point ───────────────────────────────────────────────────────────────

def build_loader_app() -> pn.viewable.Viewable:
    """Landing page with two file-path inputs + Load button; replaces itself on load."""
    data_input = pn.widgets.TextInput(
        name='1. Dataset H5  (data.h5 — patches + images)',
        placeholder='/path/to/ae_results/patches/cio/{ds}/data.h5',
        width=680,
    )
    model_input = pn.widgets.TextInput(
        name='2. Model H5  (interactive.h5 / model.h5 — latents, UMAP, predictions)',
        placeholder='/path/to/ae_results/…/interactive.h5',
        width=680,
    )
    load_btn  = pn.widgets.Button(name='Load', button_type='primary', width=100)
    status_md = pn.pane.Markdown('', width=800)
    container = pn.Column(
        pn.pane.HTML('<h2>Interactive Patch Viewer</h2>', sizing_mode='stretch_width'),
        pn.pane.HTML(
            '<p style="color:#888;">Provide both paths for two-file mode, '
            'or only the Model H5 for legacy single-file mode.</p>',
            sizing_mode='stretch_width',
        ),
        data_input,
        model_input,
        pn.Row(load_btn, status_md),
    )

    def _on_load(_):
        dp = data_input.value.strip() or None
        mp = model_input.value.strip() or None
        if not dp and not mp:
            status_md.object = '*Enter at least one H5 path.*'
            return
        for label, p in [('Dataset H5', dp), ('Model H5', mp)]:
            if p and not Path(p).exists():
                status_md.object = f'*{label} not found: `{p}`*'
                return
        status_md.object = 'Loading …'
        try:
            app = build_app(data_h5=dp, model_h5=mp)
            container.objects = [app]
        except Exception as exc:
            status_md.object = f'**Error:** `{exc}`'

    load_btn.on_click(_on_load)
    return container


if pn.state.served:
    _data_h5, _model_h5 = _get_cli_paths()
    if _data_h5 or _model_h5:
        build_app(data_h5=_data_h5, model_h5=_model_h5).servable()
    else:
        build_loader_app().servable()

if __name__ == '__main__':
    import argparse
    import socket as _socket

    ap = argparse.ArgumentParser(description='Interactive FA patch viewer.')
    ap.add_argument('h5', nargs='*', help='data.h5 [model.h5]  (0–2 files)')
    ap.add_argument('--port',  type=int, default=5006)
    ap.add_argument('--serve', action='store_true',
                    help='Bind to 0.0.0.0 for network access (others connect via IP)')
    _args = ap.parse_args()

    if len(_args.h5) >= 2:
        _data_h5, _model_h5 = _args.h5[0], _args.h5[1]
    elif len(_args.h5) == 1:
        _data_h5, _model_h5 = None, _args.h5[0]
    else:
        _data_h5, _model_h5 = None, None

    _app = (build_app(data_h5=_data_h5, model_h5=_model_h5)
            if (_data_h5 or _model_h5) else build_loader_app())

    if _args.serve:
        try:
            _s = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
            _s.connect(('8.8.8.8', 80))
            _host_ip = _s.getsockname()[0]
            _s.close()
        except Exception:
            _host_ip = '0.0.0.0'
        print(f'[view] Serving on http://{_host_ip}:{_args.port}')
        pn.serve(_app, address='0.0.0.0', port=_args.port,
                 allow_websocket_origin=['*'], show=False, autoreload=False)
    else:
        pn.serve(_app, show=True, port=_args.port, autoreload=False)
