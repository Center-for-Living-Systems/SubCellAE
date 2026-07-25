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
    df, patches_raw, images_raw, img_meta,
    images_allch, channel_names, pad_size, image_scale
      images_allch : {group_key: (C, H, W) float32}  — all channels per frame
      channel_names: ['paxillin', 'vinculin', 'zyxin', ...]
    """
    with h5py.File(path, 'r') as f:
        df          = pd.read_csv(io.StringIO(f['meta/csv'][()].decode()))
        patches_raw = f['patches/raw'][()] if 'patches/raw' in f else None
        images_raw  = f['images/raw'][()]  if 'images/raw'  in f else None
        img_meta    = (pd.read_csv(io.StringIO(f['images/meta'][()].decode()))
                       if 'images/meta' in f else None)
        pad_size    = float(f.attrs.get('pad_size', 64))
        image_scale = float(f.attrs.get('image_scale', 1.0))

        # Extra channel arrays (e.g. vinc, zyx, act)
        _abbrs = list(f.attrs.get('channels', []))
        _extra = [a for a in _abbrs if a != 'pax' and f'images/{a}' in f]
        if not _extra:
            _extra = sorted(k for k in f.get('images', {}).keys()
                            if k not in ('raw', 'meta'))
        _extra_arrs = {k: f[f'images/{k}'][()] for k in _extra}
        ch_keys = ['pax'] + list(_extra)
        channel_names = ['paxillin'] + [_CH_ABBR.get(a, a) for a in _extra]

        # Build per-group multi-channel stack
        images_allch: dict[str, np.ndarray] = {}
        if img_meta is not None and images_raw is not None:
            for _, row in img_meta.iterrows():
                fi  = int(row['frame'])
                grp = _NORM_FKEY.sub(r'_f\1', str(row['group']))
                chs = [images_raw[fi].astype(np.float32)]
                for k in _extra:
                    arr = _extra_arrs[k]
                    chs.append(arr[fi].astype(np.float32) if fi < len(arr)
                               else np.zeros_like(chs[0]))
                images_allch[grp] = np.stack(chs)   # (C, H, W)

    return (df, patches_raw, images_raw, img_meta,
            images_allch, channel_names, ch_keys, pad_size, image_scale)


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
    (df, patches_raw, images_raw, img_meta,
     images_allch, channel_names, ch_keys, pad_size, image_scale) = _read_h5(data_h5)
    n = len(df)
    patch_maxes = patches_raw.max(axis=(1, 2)) if patches_raw is not None else None
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
    px_vals  = (patches_raw.flatten().astype(np.float32)
                if patches_raw is not None else np.zeros(0, dtype=np.float32))

    fig_hist, ax = plt.subplots(figsize=(3.0, 3.0))
    ax.hist(px_vals, bins=200, color='#888888', alpha=0.75, density=True)
    ax.axvline(0, color='#4488FF', lw=1.5, ls='--', label='< 0')
    ax.axvline(1, color='#009944', lw=1.5, ls='--', label='> 1')
    ax.axvline(2, color='#FF44FF', lw=1.5, ls='--', label='> 2')
    ax.axvline(4, color='#FF4444', lw=1.5, ls='--', label='> 4')
    ax.set_xlabel('paxillin intensity')
    ax.set_ylabel('density')
    n_b = int((px_vals < 0).sum()); n_g = int((px_vals > 1).sum())
    n_m = int((px_vals > 2).sum()); n_r = int((px_vals > 4).sum())
    ax.set_title(f'pax-ch1  N={n}\n<0:{n_b}  >1:{n_g}  >2:{n_m}  >4:{n_r}',
                 fontsize=10)
    ax.legend(fontsize=8)
    fig_hist.tight_layout()

    _ch_id_str = '-'.join(ch_canvas_labels)
    dataset_info_html = pn.pane.HTML(
        f'<div style="font-size:12px;line-height:2.0;white-space:nowrap;">'
        f'<b>Dataset {ds_idx}:</b> {ds_name} &mdash; {_ch_id_str}&nbsp;&nbsp;&nbsp;'
        f'<b>Patches:</b> {n}&nbsp;&nbsp;&nbsp;'
        f'<b>Conditions:</b> {", ".join(str(c) for c in conds)}</div>',
    )
    hist_col = pn.Column(_fig_to_pane(fig_hist, dpi=100), width=320)

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
    def _get_pax_canvas(group_key: str) -> np.ndarray:
        matches = img_meta[img_meta['group'].astype(str) == group_key]
        if matches.empty:
            return np.zeros((images_raw.shape[1], images_raw.shape[2]), dtype=np.float32)
        arr = images_raw[int(matches.iloc[0]['frame'])].astype(np.float32)
        return arr if arr.ndim == 2 else arr[0]

    def _get_allch(group_key: str) -> np.ndarray | None:
        """Return (C, H, W) float32 for group_key, or None."""
        return images_allch.get(_norm_grp(group_key))

    init_group = unique_groups[0]
    init_arr   = _get_pax_canvas(init_group)
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
    side_ch_indices = list(range(1, n_ch))   # all channels except paxillin (index 0)
    side_srcs: list[ColumnDataSource] = []
    side_figs: list = []
    side_mappers: list[LinearColorMapper] = []
    _blank_canvas = np.zeros((H, W), dtype=np.float32)

    for ci in side_ch_indices:
        ch_name = ch_canvas_labels[ci] if ci < len(ch_canvas_labels) else f'ch{ci}'
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
        allch = _get_allch(group_key)
        for i, (csrc, ci) in enumerate(zip(side_srcs, side_ch_indices)):
            if allch is not None and ci < allch.shape[0]:
                cimg = _display_norm(allch[ci])
                csrc.data = dict(image=[_flip(cimg)],
                                 x=[0], y=[0],
                                 dw=[cimg.shape[1]], dh=[cimg.shape[0]])
            else:
                csrc.data = dict(image=[_flip(_blank_canvas)],
                                 x=[0], y=[0], dw=[W], dh=[H])

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
        xs, ys, ws, hs, cols, lws, las, idxs = [], [], [], [], [], [], [], []
        for (x, y, w, h, mv, di) in _rects_base['data']:
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
        arr  = _get_pax_canvas(_state['group'])
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
        arr    = _get_pax_canvas(group_key)
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
        pmax  = float(patch_maxes[idx]) if patch_maxes is not None else float('nan')
        pmean = float(row.get('mean_intensity', float('nan')))
        detail_info.object = (
            f'<span style="font-size:11px;white-space:nowrap;">'
            f'<b>Patch:</b> <code>{Path(fname).stem}</code>&nbsp;&nbsp;'
            f'<b>Cond:</b> {cond}&nbsp;&nbsp;'
            f'<b>Max:</b> {pmax:.3f}&nbsp;&nbsp;<b>Mean:</b> {pmean:.3f}</span>'
        )

        # Collect all channel arrays for this patch
        ch_arrays: list[np.ndarray] = []

        # Ch 0: paxillin from patches_raw
        if patches_raw is not None:
            ch_arrays.append(patches_raw[idx].astype(np.float32))
        else:
            ch_arrays.append(np.zeros((32, 32), dtype=np.float32))

        # Other channels: crop from full-frame allch images
        allch = _get_allch(_state['group'])
        cx = int(float(row.get('canvas_cx', 0)))
        cy = int(float(row.get('canvas_cy', 0)))
        ps = int(row.get('ps', 32))
        half = ps // 2
        for ci in side_ch_indices:
            if allch is not None and ci < allch.shape[0]:
                C, Hf, Wf = allch.shape
                y0, y1 = max(0, cy - half), min(Hf, cy + half)
                x0, x1 = max(0, cx - half), min(Wf, cx + half)
                patch = allch[ci, y0:y1, x0:x1].astype(np.float32)
                # Pad if crop hit an edge
                if patch.shape != (ps, ps):
                    padded = np.zeros((ps, ps), dtype=np.float32)
                    padded[:patch.shape[0], :patch.shape[1]] = patch
                    patch = padded
                ch_arrays.append(patch)
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
    detail_row = pn.Row(pn.Column(detail_info, width=465), dim_toggle)
    bottom_row = pn.Row(hist_col, pn.Column(detail_row, patch_pane))

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
        routes = {}
        for i, path in enumerate(args.h5):
            ds = Path(path).parent.name
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
