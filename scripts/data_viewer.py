#!/usr/bin/env python3
"""
data_viewer.py
==============
Multi-dataset interactive data viewer (data-only, no model outputs).

Layout per dataset
------------------
  ┌─────────────┬──────────────────────────┬───────────────────────────────┐
  │  Histogram  │  Paxillin canvas         │  Other-channel canvases       │
  │  (left)     │  + controls/overlays     │  (linked pan/zoom, grid only) │
  ├─────────────┴──────────────────────────┴───────────────────────────────┤
  │  Selected-patch detail — all channels in a row (click canvas to show)  │
  └────────────────────────────────────────────────────────────────────────┘

Pixel intensity overlay (blue/green/magenta/red) is paxillin-only.
Grid rect borders are drawn on all channel canvases.

Usage
-----
    # Single dataset — served at /
    python scripts/data_viewer.py cio/vinc/data.h5

    # Four datasets — served at /vinc  /ppax  /pfak  /nih3t3
    python scripts/data_viewer.py \\
        cio/vinc/data.h5  cio/ppax/data.h5 \\
        cio/pfak/data.h5  cio/nih3t3/data.h5

    python scripts/data_viewer.py cio/*/data.h5 --port 5008 --serve  # lab mode
"""
from __future__ import annotations

import argparse
import io
import re
import socket
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import panel as pn
from bokeh.events import Tap
from bokeh.models import ColumnDataSource, LinearColorMapper, Range1d
from bokeh.plotting import figure

pn.extension(sizing_mode='stretch_width')

try:
    from bokeh.palettes import gray as _bk_gray
    GRAY256 = _bk_gray(256)
except Exception:
    GRAY256 = [f'#{i:02x}{i:02x}{i:02x}' for i in range(256)]

_CH_ABBR = {
    'pax': 'paxillin', 'zyx': 'zyxin', 'act': 'actin',
    'vinc': 'vinculin', 'pfak': 'pFAK', 'ppax': 'pPax',
}
_CH_IDX = {'pax': 1, 'zyx': 2, 'act': 3, 'vinc': 0, 'ppax': 0, 'pfak': 0}
_CH_SHORT = {'pax': 'pax', 'zyx': 'zyxin', 'act': 'actin',
             'vinc': 'vinc', 'ppax': 'ppax', 'pfak': 'pfak'}
_NORM_FKEY = re.compile(r'_f0*(\d)')


# ── Data loading ──────────────────────────────────────────────────────────────

def _read_h5(path: str):
    """Load data.h5.

    Returns
    -------
    df, patch_arrays, patch_norms, images_raw, img_meta,
    channel_names, ch_keys, pad_size, image_scale, canvas_norms

      patch_arrays : {key: (N,H,W)} — flat (includes 'raw', backward-compat cio_* pax)
      patch_norms  : {norm: {ch: (N,H,W)}} — new nested per-norm per-channel patches
      canvas_norms : {norm: {ch: (M,H,W)}} — per-norm per-channel canvas images
      ch_keys      : ['pax', 'vinc'/'pfak'/'ppax', 'zyx', 'act']
    """
    import json as _json
    with h5py.File(path, 'r') as f:
        df = pd.read_csv(io.StringIO(f['meta/csv'][()].decode()))

        # ── Patches ──────────────────────────────────────────────────────────
        patch_arrays: dict[str, np.ndarray] = {}
        patch_norms:  dict[str, dict[str, np.ndarray]] = {}

        if 'patches/raw' in f:
            patch_arrays['raw'] = f['patches/raw'][()]

        if 'patches' in f:
            for key in f['patches'].keys():
                item = f[f'patches/{key}']
                if isinstance(item, h5py.Dataset):
                    # old flat structure — keep for backward compat
                    patch_arrays[key] = item[()]
                    if key != 'raw':
                        patch_norms.setdefault(key, {})['pax'] = item[()]
                elif isinstance(item, h5py.Group):
                    patch_norms[key] = {ch: item[ch][()] for ch in item.keys()}
                    if 'pax' in patch_norms[key] and key not in patch_arrays:
                        patch_arrays[key] = patch_norms[key]['pax']

        # ── Canvas images ─────────────────────────────────────────────────────
        canvas_norms: dict[str, dict[str, np.ndarray]] = {}
        images_raw = f['images/raw'][()] if 'images/raw' in f else None

        if 'images' in f:
            for key in f['images'].keys():
                if key in ('raw', 'meta', 'raw_med', 'raw_mode', 'raw_mode_prt'):
                    continue
                item = f[f'images/{key}']
                if isinstance(item, h5py.Group):
                    canvas_norms[key] = {ch: item[ch][()] for ch in item.keys()}

        # Old flat fallbacks → inject into canvas_norms
        _old_map = {
            'raw_med':      'cio_med',
            'raw_mode':     'cio_mode',
            'raw_mode_prt': 'cio_mode_prt',
        }
        for old_k, norm_k in _old_map.items():
            if f'images/{old_k}' in f and norm_k not in canvas_norms:
                canvas_norms.setdefault(norm_k, {})['pax'] = f[f'images/{old_k}'][()]
        if images_raw is not None and 'cio_inlier' not in canvas_norms:
            canvas_norms.setdefault('cio_inlier', {})['pax'] = images_raw

        # images_raw fallback
        if images_raw is None:
            images_raw = canvas_norms.get('cio_inlier', {}).get('pax')
        if images_raw is None:
            images_raw = next((v.get('pax') for v in canvas_norms.values() if 'pax' in v), None)

        img_meta = (pd.read_csv(io.StringIO(f['images/meta'][()].decode()))
                    if 'images/meta' in f else None)
        pad_size    = float(f.attrs.get('pad_size', 64))
        image_scale = float(f.attrs.get('image_scale', 1.0))

        # Channel list — from attrs or inferred from canvas/patch groups
        try:
            ch_keys_raw: list[str] = _json.loads(f.attrs['channels'])
        except Exception:
            ch_keys_raw = []
        if not ch_keys_raw:
            for v in canvas_norms.values():
                ch_keys_raw = list(v.keys()); break
        if not ch_keys_raw:
            ch_keys_raw = ['pax']

    ch_keys = ch_keys_raw
    channel_names = [_CH_ABBR.get(k, k) for k in ch_keys]

    return (df, patch_arrays, patch_norms, images_raw, img_meta,
            channel_names, ch_keys, pad_size, image_scale, canvas_norms)


# ── Display helpers ───────────────────────────────────────────────────────────

def _norm_grp(k: str) -> str:
    return _NORM_FKEY.sub(r'_f\1', k)


def _display_norm(arr: np.ndarray, pct: float = 99.9) -> np.ndarray:
    """Clip at percentile and scale to [0, 1] for display."""
    hi = float(np.percentile(arr, pct))
    if hi <= 0:
        hi = float(arr.max())
    return (np.clip(arr, 0, hi) / hi).astype(np.float32) if hi > 0 else arr.astype(np.float32)


def _flip(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.flipud(arr))


def _fig_to_pane(fig, dpi: int = 120) -> pn.pane.PNG:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return pn.pane.PNG(buf, sizing_mode='scale_width')


def _make_pixel_overlay(arr: np.ndarray, patch_mask: np.ndarray,
                        show_blue: bool, show_green: bool,
                        show_magenta: bool, show_red: bool) -> np.ndarray:
    """uint32 RGBA canvas overlay for Bokeh image_rgba (y-flipped, paxillin only)."""
    H, W = arr.shape[:2]
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    if show_blue:
        rgba[patch_mask & (arr < 0)]               = [0x44, 0x88, 0xFF, 160]
    if show_green:
        rgba[patch_mask & (arr > 1) & (arr <= 2)]  = [0x00, 0x99, 0x44, 160]
    if show_magenta:
        rgba[patch_mask & (arr > 2) & (arr <= 4)]  = [0xFF, 0x44, 0xFF, 160]
    if show_red:
        rgba[patch_mask & (arr > 4)]               = [0xFF, 0x44, 0x44, 160]
    packed = np.ascontiguousarray(rgba).view(np.uint32).reshape(H, W)
    return np.ascontiguousarray(np.flipud(packed))


def _patch_intensity_overlay(arr: np.ndarray,
                              show_blue: bool, show_green: bool,
                              show_magenta: bool, show_red: bool) -> np.ndarray | None:
    if not (show_blue or show_green or show_magenta or show_red):
        return None
    ov = np.zeros((*arr.shape[:2], 4), dtype=np.float32)
    if show_blue:
        ov[arr < 0]                = [0x44/255, 0x88/255, 1.0,      160/255]
    if show_green:
        ov[(arr > 1) & (arr <= 2)] = [0.0,      0x99/255, 0x44/255, 160/255]
    if show_magenta:
        ov[(arr > 2) & (arr <= 4)] = [1.0,      0x44/255, 1.0,      160/255]
    if show_red:
        ov[arr > 4]                = [1.0,      0x44/255, 0x44/255, 160/255]
    return ov


def _all_channel_figure(ch_arrays: list[np.ndarray],
                        ch_names:  list[np.ndarray],
                        title: str = '',
                        vmax: float = 1.0,
                        show_blue: bool = False, show_green: bool = False,
                        show_magenta: bool = False, show_red: bool = False,
                        ) -> pn.pane.PNG:
    """Render all-channel patch thumbnails. Pixel overlay on paxillin only."""
    n = len(ch_arrays)
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 2.5))
    if n == 1:
        axes = [axes]
    for i, (ax, arr, name) in enumerate(zip(axes, ch_arrays, ch_names)):
        is_pax = i == 0
        vm  = vmax if is_pax else 1.0
        disp = arr if is_pax else _display_norm(arr)
        ax.imshow(disp, cmap='gray', vmin=0, vmax=vm, interpolation='nearest')
        if is_pax:
            ov = _patch_intensity_overlay(arr, show_blue, show_green, show_magenta, show_red)
            if ov is not None:
                ax.imshow(ov, interpolation='nearest')
        ax.set_title(name, fontsize=8)
        ax.axis('off')
    if title:
        fig.suptitle(title, fontsize=8, y=1.01)
    fig.tight_layout(pad=0.3)
    return _fig_to_pane(fig)


def _rect_color(v: float, show_green: bool, show_magenta: bool, show_red: bool
                ) -> tuple[str, float, float]:
    """(border_color, line_width, line_alpha) from patch max + toggle states.
    Gray thin border is always the fallback when a category toggle is off.
    """
    if v > 4:
        return ('#FF4444', 1.5, 1.0)  if show_red     else ('#888888', 0.3, 0.4)
    if v > 2:
        return ('#FF44FF', 0.5, 0.85) if show_magenta else ('#888888', 0.3, 0.4)
    if v > 1:
        return ('#009944', 0.5, 0.75) if show_green   else ('#888888', 0.3, 0.4)
    return '#888888', 0.3, 0.4


# ── Per-dataset app builder ───────────────────────────────────────────────────

def build_app(data_h5: str, ds_idx: int = 1) -> pn.viewable.Viewable:
    print(f'[data_viewer] Loading {data_h5} …', flush=True)
    (df, patch_arrays, patch_norms, images_raw, img_meta,
     channel_names, ch_keys, pad_size, image_scale,
     canvas_norms) = _read_h5(data_h5)
    n = len(df)
    patches_raw = patch_arrays.get('raw')

    # _patch_maxes_cache: norm → pax patch maxes (for grid colouring)
    _patch_maxes_cache: dict[str, np.ndarray | None] = {}
    for k, v in patch_arrays.items():
        _patch_maxes_cache[k] = v.max(axis=(1, 2))
    for nk, ch_dict in patch_norms.items():
        if nk not in _patch_maxes_cache and 'pax' in ch_dict:
            _patch_maxes_cache[nk] = ch_dict['pax'].max(axis=(1, 2))

    _avail_norms = list(dict.fromkeys(
        list(canvas_norms.keys()) + list(patch_norms.keys())
    ))
    _default_norm = ('cio_inlier' if 'cio_inlier' in _avail_norms
                     else (_avail_norms[0] if _avail_norms else 'raw'))
    _active_norm: list[str] = [_default_norm]
    patch_maxes = _patch_maxes_cache.get(_default_norm)
    if patch_maxes is None:
        patch_maxes = _patch_maxes_cache.get('raw')

    ds_name  = Path(data_h5).parent.name
    n_ch     = len(channel_names)          # total channels (pax + extras)

    # Canvas labels: "pax-ch1", "ppax-ch0", "zyxin-ch2", "actin-ch3"
    ch_canvas_labels = [
        f'{_CH_SHORT.get(k, k)}-ch{_CH_IDX.get(k, "?")}' for k in ch_keys
    ]
    channel_names = ch_canvas_labels   # use ch-index labels everywhere

    print(f'[data_viewer] {ds_name}: {n} patches, {n_ch} channels: {channel_names}',
          flush=True)

    # ── Histogram panel ───────────────────────────────────────────────────────
    cond_col = 'condition_name' if 'condition_name' in df.columns else 'condition'
    conds    = df[cond_col].fillna('').unique() if cond_col in df.columns else []

    def _make_hist_pane(norm_key: str) -> pn.pane.PNG:
        arr = patch_norms.get(norm_key, {}).get('pax')
        if arr is None:
            arr = patch_arrays.get(norm_key)
        px_vals = arr.flatten().astype(np.float32) if arr is not None else np.zeros(0, np.float32)
        fig_h, ax = plt.subplots(figsize=(3.0, 3.0))
        ax.hist(px_vals, bins=200, color='#888888', alpha=0.75, density=True)
        ax.axvline(0, color='#4488FF', lw=1.5, ls='--', label='< 0')
        ax.axvline(1, color='#009944', lw=1.5, ls='--', label='> 1')
        ax.axvline(2, color='#FF44FF', lw=1.5, ls='--', label='> 2')
        ax.axvline(4, color='#FF4444', lw=1.5, ls='--', label='> 4')
        ax.set_xlabel('paxillin intensity')
        ax.set_ylabel('density')
        n_b = int((px_vals < 0).sum()); n_g = int((px_vals > 1).sum())
        n_m = int((px_vals > 2).sum()); n_r = int((px_vals > 4).sum())
        ax.set_title(f'{norm_key}  N={n}\n<0:{n_b}  >1:{n_g}  >2:{n_m}  >4:{n_r}',
                     fontsize=10)
        ax.legend(fontsize=8)
        fig_h.tight_layout()
        return _fig_to_pane(fig_h, dpi=100)

    # Pre-render histograms for each available normalization
    _hist_norms = list(dict.fromkeys(list(patch_norms.keys()) + list(patch_arrays.keys())))
    _hist_panes: dict[str, pn.pane.PNG] = {k: _make_hist_pane(k) for k in _hist_norms}
    hist_col = pn.Column(_hist_panes.get(_default_norm, pn.pane.Markdown('')), width=320)

    _ch_id_str = '-'.join(ch_canvas_labels)
    dataset_info_html = pn.pane.HTML(
        f'<div style="font-size:12px;line-height:2.0;white-space:nowrap;">'
        f'<b>Dataset {ds_idx}:</b> {ds_name} &mdash; {_ch_id_str}&nbsp;&nbsp;&nbsp;'
        f'<b>Patches:</b> {n}&nbsp;&nbsp;&nbsp;'
        f'<b>Conditions:</b> {", ".join(str(c) for c in conds)}</div>',
    )

    # ── Widgets ───────────────────────────────────────────────────────────────
    _W0, _W1, _W2, _W3 = 105, 140, 115, 85   # column widths: <0 | >1 | >2 | >4
    ck_blue    = pn.widgets.Checkbox(name='< 0  (blue)',       value=False, width=_W0)
    ck_green   = pn.widgets.Checkbox(name='> 1  (dark green)', value=False, width=_W1)
    ck_magenta = pn.widgets.Checkbox(name='> 2  (magenta)',    value=False, width=_W2)
    ck_red_px  = pn.widgets.Checkbox(name='> 4  (red)',        value=False, width=_W3)
    dim_toggle = pn.widgets.Toggle(name='Dim  (vmax=2)', value=False, width=110)

    ck_grid_green   = pn.widgets.Checkbox(name='> 1  green',   value=False, width=_W1)
    ck_grid_magenta = pn.widgets.Checkbox(name='> 2  magenta', value=False, width=_W2)
    ck_grid_red     = pn.widgets.Checkbox(name='> 4  red',     value=False, width=_W3)

    _last_idx: list[int | None] = [None]

    # ── Early exit: no images ─────────────────────────────────────────────────
    if images_raw is None or img_meta is None:
        return pn.Row(left_col, pn.pane.Markdown('*No canvas images in this data.h5*'))

    pg_col        = 'group' if 'group' in df.columns else cond_col
    unique_groups = img_meta['group'].astype(str).tolist() if 'group' in img_meta.columns else []
    if not unique_groups:
        return pn.Row(left_col, pn.pane.Markdown('*No image groups found.*'))

    # ── Canvas helpers ────────────────────────────────────────────────────────
    def _get_canvas(group_key: str, ch: str = 'pax') -> np.ndarray:
        matches = img_meta[img_meta['group'].astype(str) == group_key]
        norm = _active_norm[0]
        ch_map = canvas_norms.get(norm)
        if not ch_map:
            ch_map = canvas_norms.get('cio_inlier') or {}
        arr_stack = ch_map.get(ch)
        if arr_stack is None:
            arr_stack = ch_map.get('pax')
        if arr_stack is None and images_raw is not None and ch == 'pax':
            arr_stack = images_raw
        if matches.empty or arr_stack is None:
            h = images_raw.shape[1] if images_raw is not None else 1024
            w = images_raw.shape[2] if images_raw is not None else 1024
            return np.zeros((h, w), dtype=np.float32)
        arr = arr_stack[int(matches.iloc[0]['frame'])].astype(np.float32)
        return arr if arr.ndim == 2 else arr[0]

    init_group = unique_groups[0]
    init_arr   = _get_canvas(init_group, 'pax')
    H, W       = init_arr.shape
    _aspect    = H / W if W > 0 else 1.0
    _pax_w, _pax_h   = 520, int(520 * _aspect)
    _side_w, _side_h = 265, int(265 * _aspect)
    _state     = {'group': init_group, 'H': H, 'W': W}
    _rects_base: dict[str, list] = {'data': []}

    # ── Shared rect / selection sources (all canvases share the same source) ──
    rects_src = ColumnDataSource(dict(x=[], y=[], width=[], height=[],
                                      color=[], lw=[], la=[], df_idx=[]))
    sel_src   = ColumnDataSource(dict(x=[], y=[], width=[], height=[]))

    # ── Main paxillin canvas ──────────────────────────────────────────────────
    img_src     = ColumnDataSource(dict(image=[_flip(init_arr)],
                                        x=[0], y=[0], dw=[W], dh=[H]))
    overlay_src = ColumnDataSource(dict(image=[np.zeros((H, W), dtype=np.uint32)],
                                        x=[0], y=[0], dw=[W], dh=[H]))

    pax_fig = figure(
        width=_pax_w, height=_pax_h,
        x_range=Range1d(0, W), y_range=Range1d(0, H),
        title=f'{ch_canvas_labels[0]}  (click to inspect)',
        tools='tap,pan,wheel_zoom,reset',
        toolbar_location='above',
    )
    gray_mapper = LinearColorMapper(palette=GRAY256, low=-0.01, high=1.0)
    _pax_r = pax_fig.image(image='image', source=img_src,
                            x=0, y=0, dw=W, dh=H, color_mapper=gray_mapper)
    _pax_r.nonselection_glyph.global_alpha = 1.0

    def _on_dim(e):
        new_high = 2.0 if e.new else 1.0
        gray_mapper.high = new_high
        for _m in side_mappers:
            _m.high = new_high
    dim_toggle.param.watch(_on_dim, 'value')

    pax_fig.rect('x', 'y', 'width', 'height', source=rects_src,
                 fill_alpha=0, line_color='color', line_width='lw', line_alpha='la',
                 nonselection_fill_alpha=0, nonselection_line_alpha='la')
    pax_fig.rect('x', 'y', 'width', 'height', source=sel_src,
                 fill_alpha=0, line_color='white', line_width=2.5,
                 nonselection_fill_alpha=0, nonselection_line_alpha=1.0)
    _ov_r = pax_fig.image_rgba(image='image', source=overlay_src,
                                x='x', y='y', dw='dw', dh='dh')
    _ov_r.nonselection_glyph.global_alpha = 1.0
    pax_fig.xaxis.axis_label = 'column (px)'
    pax_fig.yaxis.axis_label = 'row (px)'

    # ── Side-channel canvases (linked ranges, grid only, no intensity overlay) ─
    side_ch_keys    = ch_keys[1:]          # all channels except paxillin
    side_ch_indices = list(range(1, n_ch)) # for backward compat references
    side_srcs: list[ColumnDataSource] = []
    side_figs: list = []
    side_mappers: list[LinearColorMapper] = []
    _blank_canvas = np.zeros((H, W), dtype=np.float32)

    for ci, ch_key in enumerate(side_ch_keys, start=1):
        ch_name = ch_canvas_labels[ci] if ci < len(ch_canvas_labels) else ch_key
        _src = ColumnDataSource(dict(image=[_flip(_blank_canvas)],
                                     x=[0], y=[0], dw=[W], dh=[H]))
        _fig = figure(
            width=_side_w, height=_side_h,
            x_range=pax_fig.x_range, y_range=pax_fig.y_range,
            title=ch_name,
            tools='', toolbar_location=None,
        )
        _side_mapper = LinearColorMapper(palette=GRAY256, low=0.0, high=1.0)
        side_mappers.append(_side_mapper)
        _fig.image(image='image', source=_src, x='x', y='y', dw='dw', dh='dh',
                   color_mapper=_side_mapper)
        _fig.rect('x', 'y', 'width', 'height', source=rects_src,
                  fill_alpha=0, line_color='color', line_width='lw', line_alpha='la',
                  nonselection_fill_alpha=0, nonselection_line_alpha='la')
        _fig.rect('x', 'y', 'width', 'height', source=sel_src,
                  fill_alpha=0, line_color='white', line_width=1.5,
                  nonselection_fill_alpha=0, nonselection_line_alpha=1.0)
        side_srcs.append(_src)
        side_figs.append(_fig)

    def _update_side_canvases(group_key: str):
        for csrc, ch_key in zip(side_srcs, side_ch_keys):
            cimg = _get_canvas(group_key, ch_key).astype(np.float32)
            csrc.data = dict(image=[_flip(cimg)],
                             x=[0], y=[0],
                             dw=[cimg.shape[1]], dh=[cimg.shape[0]])

    # ── Rect helpers ──────────────────────────────────────────────────────────
    def _build_rects_base(group_key: str, img_H: int) -> list:
        base = []
        for i, row in df[df[pg_col].astype(str) == group_key].iterrows():
            cx = row.get('canvas_cx', float('nan'))
            cy = row.get('canvas_cy', float('nan'))
            ps = int(row.get('ps', 32))
            if pd.isna(cx) or pd.isna(cy):
                continue
            x  = float(cx) * image_scale
            y  = (img_H - float(cy)) * image_scale
            w  = float(ps) * image_scale
            mv = float(patch_maxes[i]) if patch_maxes is not None else 0.0
            base.append((x, y, w, w, mv, i))
        return base

    def _apply_rect_colors():
        sg, sm, sr = ck_grid_green.value, ck_grid_magenta.value, ck_grid_red.value
        active_maxes = _patch_maxes_cache.get(_active_norm[0])
        xs, ys, ws, hs, cols, lws, las, idxs = [], [], [], [], [], [], [], []
        for (x, y, w, h, _mv, di) in _rects_base['data']:
            mv = float(active_maxes[di]) if active_maxes is not None else _mv
            col, lw, la = _rect_color(mv, sg, sm, sr)
            xs.append(x);    ys.append(y);    ws.append(w);   hs.append(h)
            cols.append(col); lws.append(lw); las.append(la); idxs.append(di)
        rects_src.data = dict(x=xs, y=ys, width=ws, height=hs,
                              color=cols, lw=lws, la=las, df_idx=idxs)

    # ── Pixel overlay (paxillin only) ─────────────────────────────────────────
    def _refresh_overlay():
        sb, sg, sm, sr = (ck_blue.value, ck_green.value,
                          ck_magenta.value, ck_red_px.value)
        Hc, Wc = _state['H'], _state['W']
        if not (sb or sg or sm or sr):
            overlay_src.data = dict(image=[np.zeros((Hc, Wc), dtype=np.uint32)],
                                    x=[0], y=[0], dw=[Wc], dh=[Hc])
            return
        arr  = _get_canvas(_state['group'], 'pax')
        H2, W2 = arr.shape
        mask = np.zeros((H2, W2), dtype=bool)
        for _, row in df[df[pg_col].astype(str) == _state['group']].iterrows():
            cx = row.get('canvas_cx', float('nan'))
            cy = row.get('canvas_cy', float('nan'))
            ps = int(row.get('ps', 32))
            if pd.isna(cx) or pd.isna(cy):
                continue
            half = ps // 2
            mask[max(0, int(cy)-half):min(H2, int(cy)+half),
                 max(0, int(cx)-half):min(W2, int(cx)+half)] = True
        overlay_src.data = dict(image=[_make_pixel_overlay(arr, mask, sb, sg, sm, sr)],
                                x=[0], y=[0], dw=[W2], dh=[H2])

    # ── Group loader ──────────────────────────────────────────────────────────
    def _load_group(group_key: str):
        arr    = _get_canvas(group_key, 'pax')
        Hn, Wn = arr.shape
        img_src.data = dict(image=[_flip(arr)], x=[0], y=[0], dw=[Wn], dh=[Hn])
        pax_fig.x_range.start, pax_fig.x_range.end = 0, Wn
        pax_fig.y_range.start, pax_fig.y_range.end = 0, Hn
        sel_src.data = dict(x=[], y=[], width=[], height=[])
        _state.update(group=group_key, H=Hn, W=Wn)
        _rects_base['data'] = _build_rects_base(group_key, Hn)
        _apply_rect_colors()
        _refresh_overlay()
        _update_side_canvases(group_key)

    # ── Detail panel ──────────────────────────────────────────────────────────
    detail_info = pn.pane.HTML(
        '<i style="color:#888;font-size:11px;white-space:nowrap;">Click a patch to see all channels.</i>',
    )
    patch_pane = pn.Column(pn.pane.Markdown(''), width=int(2.5 * 100 * n_ch))

    def _show_detail(idx: int):
        _last_idx[0] = idx
        row   = df.iloc[idx]
        fname = str(row.get('filename', ''))
        cond  = str(row.get('condition_name', row.get('condition', '')))
        _active_maxes = _patch_maxes_cache.get(_active_norm[0])
        pmax  = float(_active_maxes[idx]) if _active_maxes is not None else float('nan')
        pmean = float(row.get('mean_intensity', float('nan')))
        detail_info.object = (
            f'<span style="font-size:11px;white-space:nowrap;">'
            f'<b>Patch:</b> <code>{Path(fname).stem}</code>&nbsp;&nbsp;'
            f'<b>Cond:</b> {cond}&nbsp;&nbsp;'
            f'<b>Max:</b> {pmax:.3f}&nbsp;&nbsp;<b>Mean:</b> {pmean:.3f}</span>'
        )

        # Collect all channel arrays for this patch
        ch_arrays: list[np.ndarray] = []

        norm = _active_norm[0]
        norm_ch_patches = patch_norms.get(norm, {})
        ps = int(row.get('ps', 32))
        for ch_key in ch_keys:
            ch_patch_arr = norm_ch_patches.get(ch_key)
            if ch_patch_arr is None and ch_key == 'pax':
                ch_patch_arr = (patch_arrays.get(norm)
                                or patch_arrays.get('cio_inlier')
                                or patch_arrays.get('raw'))
            if ch_patch_arr is not None and idx < len(ch_patch_arr):
                ch_arrays.append(ch_patch_arr[idx].astype(np.float32))
            else:
                ch_arrays.append(np.zeros((ps, ps), dtype=np.float32))

        patch_pane.objects = [_all_channel_figure(
            ch_arrays, channel_names,
            title=Path(fname).stem,
            vmax=2.0 if dim_toggle.value else 1.0,
            show_blue=ck_blue.value, show_green=ck_green.value,
            show_magenta=ck_magenta.value, show_red=ck_red_px.value,
        )]

    def _refresh_detail(_=None):
        if _last_idx[0] is not None:
            _show_detail(_last_idx[0])

    # ── Patch normalization selector (only shown when >1 normalization exists) ──
    _norm_keys = list(dict.fromkeys(list(canvas_norms.keys()) + list(patch_norms.keys())))
    _norm_labels = {'raw': 'raw', 'cio_inlier': 'CIO inlier',
                    'cio_med': 'CIO median', 'cio_mode': 'CIO mode',
                    'cio_mode_prt': 'CIO mode-prt'}
    norm_select = pn.widgets.RadioButtonGroup(
        options={_norm_labels.get(k, k): k for k in _norm_keys},
        value=_default_norm,
        width=300,
    )

    def _on_norm_change(event):
        _active_norm[0] = event.new
        hist_col.objects = [_hist_panes.get(event.new, pn.pane.Markdown(''))]
        _apply_rect_colors()
        arr = _get_canvas(_state['group'], 'pax')
        img_src.data = dict(image=[_flip(arr)], x=[0], y=[0],
                            dw=[arr.shape[1]], dh=[arr.shape[0]])
        _update_side_canvases(_state['group'])
        _refresh_overlay()
        _refresh_detail()

    if len(_norm_keys) > 1:
        norm_select.param.watch(_on_norm_change, 'value')

    # ── Wire watchers ─────────────────────────────────────────────────────────
    for ck in [ck_blue, ck_green, ck_magenta, ck_red_px]:
        ck.param.watch(lambda _: _refresh_overlay(), 'value')
        ck.param.watch(_refresh_detail, 'value')
    for ck in [ck_grid_green, ck_grid_magenta, ck_grid_red]:
        ck.param.watch(lambda _: _apply_rect_colors(), 'value')
    dim_toggle.param.watch(_refresh_detail, 'value')

    # ── Canvas tap handler ────────────────────────────────────────────────────
    def _on_tap(event: Tap):
        H_cur  = _state['H']
        cx_t   = event.x / image_scale
        cy_t   = (H_cur - event.y) / image_scale
        xs_bk  = np.array(rects_src.data['x'],      dtype=float)
        ys_bk  = np.array(rects_src.data['y'],      dtype=float)
        df_idx = np.array(rects_src.data['df_idx'], dtype=int)
        if len(xs_bk) == 0:
            return
        cx_arr  = xs_bk / image_scale
        cy_arr  = (H_cur - ys_bk) / image_scale
        near_i  = int(np.argmin((cx_arr - cx_t)**2 + (cy_arr - cy_t)**2))
        near_df = int(df_idx[near_i])
        sel_src.data = dict(
            x=[xs_bk[near_i]], y=[ys_bk[near_i]],
            width=[rects_src.data['width'][near_i]],
            height=[rects_src.data['height'][near_i]],
        )
        _show_detail(near_df)

    pax_fig.on_event(Tap, _on_tap)

    # ── Initial load ──────────────────────────────────────────────────────────
    _rects_base['data'] = _build_rects_base(init_group, H)
    _apply_rect_colors()
    _update_side_canvases(init_group)

    # ── Layout ────────────────────────────────────────────────────────────────
    img_select = pn.widgets.Select(
        name='Image', options=unique_groups, value=init_group, width=760,
    )
    img_select.param.watch(lambda e: _load_group(e.new), 'value')

    _lbl_w = 90
    overlay_row = pn.Row(
        pn.pane.HTML('<b style="font-size:11px;">Pax overlay:</b>', width=_lbl_w),
        ck_blue, ck_green, ck_magenta, ck_red_px,
        pn.layout.HSpacer(),
    )
    grid_row = pn.Row(
        pn.pane.HTML('<b style="font-size:11px;">Grid colors:</b>', width=_lbl_w),
        pn.pane.HTML('<span style="font-size:11px;color:#888;">■ gray</span>', width=_W0),
        ck_grid_green, ck_grid_magenta, ck_grid_red,
        pn.pane.HTML('<span style="font-size:11px;color:#888;">&nbsp;&nbsp;based on pax patch intensity max</span>'),
        pn.layout.HSpacer(),
    )

    # ── Info bar: dataset summary (left) + selected patch detail (right) ────────
    info_bar = pn.Row(dataset_info_html, pn.layout.HSpacer())

    # ── Top row: paxillin (bigger, left) + controls + side canvases (right) ──
    pax_col   = pn.Column(pn.pane.Bokeh(pax_fig), width=_pax_w + 10)
    side_panes = [pn.pane.Bokeh(f) for f in side_figs]
    side_row   = pn.Row(*side_panes) if side_panes else pn.Row()
    right_col  = pn.Column(img_select, overlay_row, grid_row, side_row)
    canvas_row = pn.Row(pax_col, right_col)

    # ── Bottom row: histogram (left) + patch thumbnails (right) ───────────────
    norm_row = (pn.Row(
                    pn.pane.HTML('<b style="font-size:11px;">Patch norm:</b>', width=90),
                    norm_select,
                ) if len(_norm_keys) > 1 else pn.Row())
    detail_row = pn.Row(pn.Column(detail_info, width=465), dim_toggle)
    bottom_row = pn.Row(hist_col, pn.Column(norm_row, detail_row, patch_pane))

    return pn.Column(info_bar, canvas_row, bottom_row)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('h5', nargs='+', help='Path(s) to data.h5 file(s)')
    ap.add_argument('--port',  type=int, default=5008)
    ap.add_argument('--serve', action='store_true',
                    help='Bind to 0.0.0.0 for network access (lab-server mode)')
    args = ap.parse_args()

    if len(args.h5) == 1:
        routes = {'/': lambda h=args.h5[0]: build_app(h, ds_idx=1)}
        print(f'[data_viewer] http://localhost:{args.port}/  →  {args.h5[0]}')
    else:
        # Build route names; prepend grandparent when ds names collide
        ds_names = [Path(p).parent.name for p in args.h5]
        from collections import Counter
        dup = {k for k, v in Counter(ds_names).items() if v > 1}
        route_names = [
            f"{Path(p).parent.parent.name}_{Path(p).parent.name}" if Path(p).parent.name in dup
            else Path(p).parent.name
            for p in args.h5
        ]
        routes = {}
        for i, (path, ds) in enumerate(zip(args.h5, route_names)):
            routes[f'/{ds}'] = (lambda h=path, idx=i+1: build_app(h, ds_idx=idx))
            print(f'[data_viewer] http://localhost:{args.port}/{ds}  →  {path}')

    if args.serve:
        try:
            _s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            _s.connect(('8.8.8.8', 80))
            host_ip = _s.getsockname()[0]
            _s.close()
        except Exception:
            host_ip = '0.0.0.0'
        print(f'[data_viewer] Lab mode — {host_ip}:{args.port}')
        pn.serve(routes, address='0.0.0.0', port=args.port,
                 allow_websocket_origin=['*'], show=False, autoreload=False)
    else:
        pn.serve(routes, port=args.port, show=False, autoreload=False)


if __name__ == '__main__':
    main()
