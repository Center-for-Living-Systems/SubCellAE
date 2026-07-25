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

_CH_IDX   = {'pax': 1, 'zyx': 2, 'act': 3, 'vinc': 0, 'ppax': 0, 'pfak': 0}
_CH_SHORT  = {'pax': 'pax', 'zyx': 'zyxin', 'act': 'actin',
               'vinc': 'vinc', 'ppax': 'ppax', 'pfak': 'pfak'}

def _ch_label(key: str) -> str:
    return f'{_CH_SHORT.get(key, key)}-ch{_CH_IDX.get(key, "?")}'

# Grayscale palette: index 0 → black, index 255 → white
try:
    from bokeh.palettes import gray as _bk_gray
    GRAY256 = _bk_gray(256)
except Exception:
    GRAY256 = [f'#{i:02x}{i:02x}{i:02x}' for i in range(256)]


def _label_color(label: str, color_map: dict) -> str:
    return color_map.get(str(label), FALLBACK)


def _max_intensity_color(v: float) -> tuple[str, float, float]:
    """Return (color, line_width, line_alpha) for a patch rect border."""
    if v > 4:  return '#FF4444', 1.5, 1.0   # red, thicker
    if v > 2:  return '#FF44FF', 0.5, 0.85  # magenta, thin
    return '#000000', 0.0, 0.0              # not shown


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

    Returns (df_data, patches_raw, images_raw, img_meta, pad, scale,
             extra_ch_images, ch_keys).
    """
    import json as _json
    with h5py.File(path, 'r') as f:
        df_data     = pd.read_csv(io.StringIO(f['meta/csv'][()].decode()))
        patches_raw = f['patches/raw'][()] if 'patches/raw' in f else None
        images_raw  = f['images/raw'][()]  if 'images/raw'  in f else None
        img_meta    = (pd.read_csv(io.StringIO(f['images/meta'][()].decode()))
                       if 'images/meta' in f else None)
        pad_size    = float(f.attrs.get('pad_size', 64))
        image_scale = float(f.attrs.get('image_scale', 1.0))
        ch_keys_raw = f.attrs.get('channels', '["pax"]')
        ch_keys: list[str] = _json.loads(ch_keys_raw) if isinstance(ch_keys_raw, str) else list(ch_keys_raw)
        extra_ch_images: dict[str, np.ndarray] = {}
        for k in ch_keys[1:]:  # skip first (pax = images/raw)
            dsk = f'images/{k}'
            if dsk in f:
                extra_ch_images[k] = f[dsk][()]
    return df_data, patches_raw, images_raw, img_meta, pad_size, image_scale, extra_ch_images, ch_keys


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
        df_data, pr_data, im_data, imm_data, pad_d, scale_d, extra_ch_images, ch_keys = \
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

        return df, patches_raw, patches_recon, images_raw, img_meta, pad_size, image_scale, result_dir, plots, extra_ch_images, ch_keys

    elif model_h5:
        print(f'[view] Loading (single-file) {model_h5}')
        nine = _read_h5_model(model_h5)
        return nine + ({}, ['pax'])

    elif data_h5:
        print(f'[view] Loading (data-only) {data_h5}')
        df_data, patches_raw, images_raw, img_meta, pad_size, image_scale, extra_ch_images, ch_keys = _read_h5_data(data_h5)
        return df_data, patches_raw, None, images_raw, img_meta, pad_size, image_scale, Path(''), {}, extra_ch_images, ch_keys

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
                  title: str = '', vmax: float = 1.0) -> pn.pane.PNG:
    panels = [('Raw', raw)] + ([('Recon', recon)] if recon is not None else [])
    fig, axes = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 2.6))
    if len(panels) == 1:
        axes = [axes]
    for ax, (lbl, arr) in zip(axes, panels):
        ax.imshow(arr, cmap='gray', vmin=0, vmax=vmax, interpolation='nearest')
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
     images_raw, img_meta, pad_size, image_scale, result_dir, plots,
     extra_ch_images, ch_keys) = load_sources(data_h5, model_h5)
    n = len(df)

    pax_key   = ch_keys[0] if ch_keys else 'pax'
    pax_label = _ch_label(pax_key)
    print(f'[view]   {n} patches, image_scale={image_scale}')
    patch_maxes: np.ndarray | None = (
        patches_raw.max(axis=(1, 2)) if patches_raw is not None else None
    )
    patch_l1s: np.ndarray | None = (
        np.abs(patches_raw).mean(axis=(1, 2)).astype(np.float32)
        if patches_raw is not None else None
    )
    if patch_l1s is not None:
        _l1_sorted = np.sort(patch_l1s)
        _l1_pcts   = {p: float(np.percentile(patch_l1s, p))
                      for p in [10, 25, 50, 75, 90]}
    else:
        _l1_sorted = None
        _l1_pcts   = None


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
        ax.axvline(0, color='#4488FF', lw=1.6, ls='--', label='< 0  (blue)')
        ax.axvline(1, color='#009944', lw=1.6, ls='--', label='> 1  (dark green)')
        ax.axvline(2, color='#FF44FF', lw=1.6, ls='--', label='> 2  (magenta)')
        ax.axvline(4, color='#FF4444', lw=1.6, ls='--', label='> 4  (red)')
        ax.set_xlabel('pixel intensity')
        ax.set_ylabel('density')
        n_blue    = int((px_vals < 0).sum())
        n_green   = int((px_vals > 1).sum())
        n_magenta = int((px_vals > 2).sum())
        n_red     = int((px_vals > 4).sum())
        ax.set_title(
            f'All patches  N={n}  ({len(px_vals):,} px)\n'
            f'<0: {n_blue}  >1: {n_green}  >2: {n_magenta}  >4: {n_red}',
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

        # UMAP filter toggle
        umap_filter = pn.widgets.RadioButtonGroup(
            options=['All patches', 'Labeled only'], value='All patches', width=230,
        )
        _umap_all_data = {k: np.array(v) for k, v in umap_data.items()}
        if 'annotation_label' in df.columns:
            labeled_mask = df['annotation_label'].fillna(-1).values >= 0
        else:
            labeled_mask = np.ones(n, dtype=bool)
        _umap_labeled_data = {k: v[labeled_mask] for k, v in _umap_all_data.items()}

        def _on_umap_filter(event):
            if event.new == 'Labeled only':
                umap_src.data = dict(_umap_labeled_data)
            else:
                umap_src.data = dict(_umap_all_data)
        umap_filter.param.watch(_on_umap_filter, 'value')

        # Single big dot on UMAP -- updated when user clicks the image panel
        highlight_src = ColumnDataSource({'x': [], 'y': [], 'color': []})

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

        # Highlighted point (from image click) -- drawn on top with label color
        p_umap.scatter(
            'x', 'y', source=highlight_src, marker='circle',
            fill_color='color', line_color='white',
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

        left_col = pn.Column(
            pn.Row(pn.pane.Bokeh(color_select), umap_filter),
            pn.pane.Bokeh(p_umap),
            width=540,
        )
    # end if/else data_only

    # ── Outlier highlight checkboxes (shared with canvas and detail panel) ───────
    dim_toggle = pn.widgets.Toggle(name='Dim  (vmax=2)', value=False, width=110)
    _last_detail_idx: list[int | None] = [None]  # tracks last patch shown in detail

    # ── Full image Bokeh figure (Direction B) ─────────────────────────────────
    has_images = (images_raw is not None and img_meta is not None) or has_old_images

    # Placeholders updated inside the has_images block
    rects_src = sel_src = img_fig = img_pane = img_select_widget = None
    _state: dict = {}

    # Extra channel collections (populated inside has_images block if applicable)
    extra_ch_figs: dict = {}
    extra_ch_srcs: dict = {}
    extra_ch_mappers: list = []
    extra_ch_panes: list = []

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
            _unique_groups_set = set(unique_groups)
            def _get_canvas(group_key: str) -> np.ndarray | None:
                frame = _get_frame(images_raw, img_meta, group_key)
                if frame is None:
                    print(f'[view] WARN: no image for group {group_key!r}', flush=True)
                    return None
                return _norm_image(frame)
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

        if '_unique_groups_set' not in dir():
            _unique_groups_set = set(unique_groups)

        img_options = {f"{grp_to_cond.get(g, '?')} | {g}": g
                       for g in unique_groups}

        init_group = unique_groups[0]
        init_arr   = _get_canvas(init_group)
        if init_arr is None:
            raise RuntimeError(f'Cannot load initial canvas for group {init_group!r}')
        H, W       = init_arr.shape[:2]

        # Image data source
        img_src = ColumnDataSource(dict(
            image=[_flip_for_bokeh(init_arr)],
            x=[0], y=[0], dw=[W], dh=[H],
        ))

        # Patch rectangle source (coloured by patch max intensity)
        rects_src = ColumnDataSource(dict(
            x=[], y=[], width=[], height=[], color=[], lw=[], la=[], df_idx=[],
        ))
        # Selected-patch white-border highlight
        sel_src = ColumnDataSource(dict(x=[], y=[], width=[], height=[]))

        # Figure
        img_fig = figure(
            width=520, height=520,
            x_range=Range1d(0, W),
            y_range=Range1d(0, H),
            title=f'{pax_label}  (click a patch to highlight on UMAP)',
            tools='tap,pan,wheel_zoom,reset',
            toolbar_location='above',
        )
        gray_mapper = LinearColorMapper(palette=GRAY256, low=-0.01, high=1.0)
        _img_r = img_fig.image(
            image='image', source=img_src,
            x=0, y=0, dw=W, dh=H,
            color_mapper=gray_mapper,
        )
        _img_r.nonselection_glyph.global_alpha = 1.0
        def _on_dim_toggle(e):
            setattr(gray_mapper, 'high', 2.0 if e.new else 1.0)
            for _m in extra_ch_mappers:
                setattr(_m, 'high', 2.0 if e.new else 1.0)
        dim_toggle.param.watch(_on_dim_toggle, 'value')
        img_fig.rect(
            'x', 'y', 'width', 'height', source=rects_src,
            fill_alpha=0, line_color='color', line_width='lw', line_alpha='la',
            nonselection_fill_alpha=0, nonselection_line_alpha='la',
        )
        img_fig.rect(
            'x', 'y', 'width', 'height', source=sel_src,
            fill_alpha=0, line_color='white', line_width=2.5,
            nonselection_fill_alpha=0, nonselection_line_alpha=1.0,
        )
        img_fig.xaxis.axis_label = 'column (px)'
        img_fig.yaxis.axis_label = 'row (px)'

        # Helper: build rect data for a group (in Bokeh flipped-y coordinates)
        def _rects_for_group(group_key: str, img_H: int) -> dict:
            mask = df[pg_col].astype(str) == group_key
            sub  = df[mask]
            xs, ys, ws, hs, cols, lws, las, idxs = [], [], [], [], [], [], [], []
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
                if patch_maxes is not None:
                    col, lw, la = _max_intensity_color(float(patch_maxes[i]))
                else:
                    col, lw, la = _label_color(str(row.get('fa_pred', '')), FA_COLOR_MAP), 0.5, 0.75
                cols.append(col)
                lws.append(lw)
                las.append(la)
                idxs.append(i)
            return dict(x=xs, y=ys, width=ws, height=hs, color=cols, lw=lws, la=las, df_idx=idxs)

        _state.update(group=init_group, H=H, W=W)
        rects_src.data = _rects_for_group(init_group, H)

        def _load_group(group_key: str) -> None:
            arr = _get_canvas(group_key)
            if arr is None:
                print(f'[view] WARN: skipping _load_group for missing group {group_key!r}', flush=True)
                return
            Hn, Wn  = arr.shape[:2]
            img_src.data = dict(
                image=[_flip_for_bokeh(arr)],
                x=[0], y=[0], dw=[Wn], dh=[Hn],
            )
            img_fig.x_range.start, img_fig.x_range.end = 0, Wn
            img_fig.y_range.start, img_fig.y_range.end = 0, Hn
            rects_src.data = _rects_for_group(group_key, Hn)
            sel_src.data   = dict(x=[], y=[], width=[], height=[])
            highlight_src.data = dict(x=[], y=[], color=[])
            _state.update(group=group_key, H=Hn, W=Wn)
            # Update extra channel canvases
            if extra_ch_images and img_meta is not None:
                grp_meta = img_meta[img_meta['group'].astype(str) == group_key]
                if not grp_meta.empty:
                    fr = int(grp_meta.iloc[0]['frame'])
                    for k, ch_src in extra_ch_srcs.items():
                        ch_arr = _norm_image(extra_ch_images[k][fr])
                        ch_src.data = dict(
                            image=[_flip_for_bokeh(ch_arr)],
                            x=[0], y=[0], dw=[Wn], dh=[Hn],
                        )

        img_select_widget = pn.widgets.Select(
            name='Image', options=img_options,
            value=init_group, width=420,
        )
        img_select_widget.param.watch(lambda e: _load_group(e.new), 'value')
        img_pane = pn.pane.Bokeh(img_fig)

        # ── Extra channel Bokeh canvases ──────────────────────────────────────
        if extra_ch_images and img_meta is not None:
            init_grp_meta = img_meta[img_meta['group'].astype(str) == init_group]
            init_frame_row = int(init_grp_meta.iloc[0]['frame']) if not init_grp_meta.empty else 0
            for k, ch_frames in extra_ch_images.items():
                init_ch_arr = _norm_image(ch_frames[init_frame_row])
                ch_src = ColumnDataSource(dict(
                    image=[_flip_for_bokeh(init_ch_arr)],
                    x=[0], y=[0], dw=[W], dh=[H],
                ))
                ch_mapper = LinearColorMapper(palette=GRAY256, low=-0.01, high=1.0)
                ch_fig = figure(
                    width=340, height=340,
                    x_range=img_fig.x_range,
                    y_range=img_fig.y_range,
                    title=_ch_label(k),
                    tools='pan,wheel_zoom,reset',
                    toolbar_location='above',
                )
                _r = ch_fig.image(
                    image='image', source=ch_src,
                    x=0, y=0, dw=W, dh=H,
                    color_mapper=ch_mapper,
                )
                _r.nonselection_glyph.global_alpha = 1.0
                ch_fig.rect(
                    'x', 'y', 'width', 'height', source=sel_src,
                    fill_alpha=0, line_color='white', line_width=2.5,
                    nonselection_fill_alpha=0, nonselection_line_alpha=1.0,
                )
                extra_ch_figs[k] = ch_fig
                extra_ch_srcs[k] = ch_src
                extra_ch_mappers.append(ch_mapper)
                extra_ch_panes.append(pn.pane.Bokeh(ch_fig))

    # ── Shared detail panel (bottom bar) ──────────────────────────────────────
    pred_md   = pn.pane.HTML(
        '<i style="color:#888;">Hover the UMAP for a quick patch preview.  '
        'Tap the UMAP or click a patch in the canvas for full details.</i>',
        width=300,
    )
    patch_col     = pn.Column(pn.pane.Markdown(''), width=300)
    side_patch_col = pn.Column(width=165)

    def _show_detail(idx: int) -> None:
        _last_detail_idx[0] = idx
        row   = df.iloc[idx]
        fa    = str(row.get('fa_pred',  '--'))
        pos   = str(row.get('pos_pred', '--'))
        fname = str(row.get('filename', ''))
        cond  = str(row.get('condition_name', row.get('condition', '')))
        fa_color  = FA_COLOR_MAP.get(fa,  FALLBACK)
        pos_color = POS_COLOR_MAP.get(pos, FALLBACK)
        if patch_l1s is not None:
            l1_val = float(patch_l1s[idx])
            l1_pct = float(np.searchsorted(_l1_sorted, l1_val) / len(_l1_sorted) * 100)
            l1_line = (
                f'<b>L1:</b> {l1_val:.4f}'
                f'<span style="color:#888;font-size:11px;">'
                f'  →  <b>p{l1_pct:.0f}</b>'
                f'&nbsp;&nbsp;|&nbsp;&nbsp;'
                f'p10={_l1_pcts[10]:.4f}'
                f'&nbsp;p25={_l1_pcts[25]:.4f}'
                f'&nbsp;p50={_l1_pcts[50]:.4f}'
                f'&nbsp;p75={_l1_pcts[75]:.4f}'
                f'&nbsp;p90={_l1_pcts[90]:.4f}'
                f'</span>'
            )
        else:
            l1_line = ''
        pred_md.object = (
            f'<div style="font-size:12px;line-height:2;">'
            f'<b>Patch:</b> <code>{Path(fname).stem}</code><br>'
            f'<b>Condition:</b> {cond}<br>'
            + (f'{l1_line}<br>' if l1_line else '')
            + f'<i>Prediction by</i> <code>{model_name}</code>:<br>'
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
                vmax=2.0 if dim_toggle.value else 1.0,
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

        # Side patch column: per-channel thumbnails from extra_ch_images
        if extra_ch_images and img_meta is not None:
            _pg_col = 'patch_group' if 'patch_group' in df.columns else 'group'
            row_df  = df.iloc[idx]
            group   = str(row_df.get(_pg_col, ''))
            grp_meta = img_meta[img_meta['group'].astype(str) == group]
            if grp_meta.empty:
                side_patch_col.objects = []
            else:
                frame_row = int(grp_meta.iloc[0]['frame'])
                cx = row_df.get('canvas_cx', np.nan)
                cy = row_df.get('canvas_cy', np.nan)
                ps_val = int(row_df.get('ps', 32))
                if pd.isna(cx) or pd.isna(cy):
                    side_patch_col.objects = []
                else:
                    ch_panes = []
                    for k, frames in extra_ch_images.items():
                        if frame_row >= len(frames):
                            print(f'[view] WARN: frame_row {frame_row} out of range for channel {k} ({len(frames)} frames)', flush=True)
                            continue
                        frame  = _norm_image(frames[frame_row])
                        cx_i   = int(round(float(cx)))
                        cy_i   = int(round(float(cy)))
                        half   = ps_val // 2
                        patch  = frame[max(0, cy_i - half):cy_i + half,
                                       max(0, cx_i - half):cx_i + half]
                        if patch.size == 0:
                            print(f'[view] WARN: empty patch crop for channel {k} at cx={cx_i} cy={cy_i}', flush=True)
                            continue
                        fig_s, ax_s = plt.subplots(figsize=(1.56, 1.56))
                        ax_s.imshow(patch, cmap='gray', vmin=0,
                                    vmax=2.0 if dim_toggle.value else 1.0)
                        ax_s.set_title(_ch_label(k), fontsize=8)
                        ax_s.axis('off')
                        fig_s.tight_layout(pad=0.2)
                        ch_panes.append(_fig_to_pane(fig_s))
                    side_patch_col.objects = ch_panes

    # Re-render detail panel when checkboxes change (so overlay updates live)
    def _refresh_detail(_=None):
        if _last_detail_idx[0] is not None:
            _show_detail(_last_detail_idx[0])

    dim_toggle.param.watch(_refresh_detail, 'value')

    # ── Direction A: UMAP tap → detail + canvas highlight ─────────────────────
    def _on_umap_tap(attr, old, new):
        if not new:
            return
        idx = int(umap_src.data['idx'][new[0]])
        _show_detail(idx)

        if not has_images:
            return
        row    = df.iloc[idx]
        pg_val = str(row.get(pg_col, ''))
        cx     = row.get('canvas_cx', np.nan)
        cy     = row.get('canvas_cy', np.nan)
        ps     = float(row.get('ps', 32))

        if not pd.isna(cx) and not pd.isna(cy):
            if pg_val and pg_val not in _unique_groups_set:
                print(f'[view] WARN: UMAP patch group {pg_val!r} not in image data — skipping canvas update', flush=True)
                return
            # Load group first so _state['H'] is correct before computing bx/by
            if pg_val and pg_val != _state.get('group'):
                _load_group(pg_val)
                img_select_widget.value = pg_val
            H_cur = _state['H']
            bx = float(cx) * image_scale
            by = (H_cur - float(cy)) * image_scale
            w  = ps * image_scale
            sel_src.data = dict(x=[bx], y=[by], width=[w], height=[w])
            # Auto-pan canvas — equal x/y span so aspect ratio is preserved.
            # Clamp end first, then slide start back to keep span = 2*margin.
            margin = w * 6
            W_sc = _state['W'] * image_scale
            H_sc = _state['H'] * image_scale

            xe = min(W_sc, bx + margin)
            xs = max(0.0,  xe - 2 * margin)
            ye = min(H_sc, by + margin)
            ys = max(0.0,  ye - 2 * margin)

            img_fig.x_range.start, img_fig.x_range.end = xs, xe
            img_fig.y_range.start, img_fig.y_range.end = ys, ye

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

            # Big dot on UMAP colored by FA label (model mode only)
            if not data_only:
                row      = df.iloc[near_df]
                umap_x   = float(row.get('UMAP_1', row.get('umap_1', 0)) or 0)
                umap_y   = float(row.get('UMAP_2', row.get('umap_2', 0)) or 0)
                dot_color = _label_color(str(row.get('fa_pred', '')), FA_COLOR_MAP)
                highlight_src.data = dict(x=[umap_x], y=[umap_y], color=[dot_color])

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
        canvas_col = pn.Column(img_select_widget,
                               pn.Row(pn.layout.HSpacer(), dim_toggle),
                               img_pane, width=540)
    else:
        canvas_col = pn.pane.Markdown(
            '*Full image data not in this HDF5.*\n\n'
            'Re-pack with `--image-scale 1.0` (default) to include canvas images.',
            width=540,
        )

    src_label = (Path(model_h5).name if model_h5 else '') + \
                (' + ' + Path(data_h5).name if data_h5 else '')
    header = pn.pane.HTML(
        f'<h2>Interactive Patch Viewer &nbsp;·&nbsp; '
        f'<code>{src_label}</code>'
        f' &nbsp;·&nbsp; <code>{model_name}</code></h2>',
        sizing_mode='stretch_width',
    )
    # Nest left_col + canvas + extra-channel row in their own Column so the
    # extra channels never shift when detail_col or side_patch_col grow.
    left_canvas_col = pn.Column(
        pn.Row(left_col, pn.Spacer(width=12), canvas_col),
        *(([pn.Row(*extra_ch_panes)]) if extra_ch_panes else []),
    )
    return pn.Column(
        header,
        pn.Row(left_canvas_col, pn.Spacer(width=12, sizing_mode='fixed'), detail_col,
               pn.Spacer(width=8, sizing_mode='fixed'), side_patch_col),
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
