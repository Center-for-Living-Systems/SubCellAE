"""
label_patches.py
================
Manual patch labelling tool.  Reads the same interactive.h5 produced by
pack_interactive_h5.py.  Shows one full canvas image at a time with all patch
boxes drawn as a grid.  Click a label button to set the active label, then
click any patch box on the canvas to assign that label to the patch.

Labels are stored as { filename → label } and written to a CSV on "Finish".

Usage
-----
    python scripts/label_patches.py path/to/interactive.h5
    python scripts/label_patches.py path/to/interactive.h5 --out my_labels.csv
    python scripts/label_patches.py path/to/interactive.h5 --port 5007
"""

from __future__ import annotations

import argparse
import io
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import panel as pn
import tifffile
from bokeh.events import DoubleTap, Tap
from bokeh.models import ColumnDataSource, LinearColorMapper, Range1d
from bokeh.plotting import figure

from subcellae.utils.label_colors import (
    classification_label_to_color as FA_COLOR_MAP,
)

pn.extension(sizing_mode='stretch_width')

# Labels available in this tool
LABEL_OPTIONS = [
    "Nascent Adhesion",
    "focal complex",
    "focal adhesion",
    "fibrillar adhesion",
    "No adhesion",
]
UNLABELED_COLOR = "#555555"

try:
    from bokeh.palettes import gray as _bk_gray
    GRAY256 = _bk_gray(256)
except Exception:
    GRAY256 = [f'#{i:02x}{i:02x}{i:02x}' for i in range(256)]


# ── HDF5 loading ──────────────────────────────────────────────────────────────

def load_h5(path: str):
    import json as _json
    _ABBR = {'pax': 'paxillin', 'zyx': 'zyxin', 'act': 'actin',
             'vinc': 'vinculin', 'pfak': 'pFAK', 'ppax': 'pPax'}
    with h5py.File(path, 'r') as f:
        df         = pd.read_csv(io.StringIO(f['meta/csv'][()].decode()))
        images_raw = f['images/raw'][()]    if 'images/raw'  in f else None
        img_meta   = (pd.read_csv(io.StringIO(f['images/meta'][()].decode()))
                      if 'images/meta' in f else None)
        patches_allch = f['patches/allch'][()] if 'patches/allch' in f else None
        images_allch: dict = {}
        channel_names = None

        if 'images/allch' in f:
            # pack_interactive_h5 / pack_labeler_from_prep format
            for gname in f['images/allch']:
                images_allch[gname] = f['images/allch'][gname][()]
            _cn = f.attrs.get('channel_names', None)
            channel_names = _json.loads(_cn) if _cn else None
        elif img_meta is not None and images_raw is not None:
            # pack_patches_label_h5 format: separate per-channel frame arrays
            # Preserve order from attrs['channels'] so name ↔ image index stays consistent
            _abbrs = list(f.attrs.get('channels', []))
            _extra = [a for a in _abbrs if a != 'pax' and f'images/{a}' in f]
            if not _extra:  # fallback if attrs missing
                _extra = sorted(k for k in f['images'].keys() if k not in ('raw', 'meta'))
            _extra_arrs = {k: f[f'images/{k}'][()] for k in _extra}
            for _, row in img_meta.iterrows():
                fi  = int(row['frame'])
                grp = str(row['group'])
                chs = ([images_raw[fi].astype(np.float32)] +
                       [_extra_arrs[k][fi].astype(np.float32) for k in _extra])
                images_allch[grp] = np.stack(chs)
            channel_names = ['paxillin'] + [_ABBR.get(a, a) for a in _extra]

        pad_size    = float(f.attrs.get('pad_size', 64))
        image_scale = float(f.attrs.get('image_scale', 1.0))
        result_dir  = Path(str(f.attrs.get('result_dir', '')))
    return df, images_raw, img_meta, patches_allch, images_allch, channel_names, pad_size, image_scale, result_dir


# ── App ───────────────────────────────────────────────────────────────────────

def build_labeler(h5_path: str, location: str = '') -> pn.viewable.Viewable:
    df, images_raw, img_meta, patches_allch, images_allch, _loaded_ch_names, _, image_scale, result_dir = load_h5(h5_path)

    if _loaded_ch_names:
        channel_names = _loaded_ch_names
    else:
        # Fallback: hard-code ch1=paxillin, ch2=zyxin, ch3=actin; detect ch0 from path
        _path_str = (str(h5_path) + ' ' + str(result_dir)).lower()
        _ch0 = next((kw for kw in ('vinc', 'pfak', 'ppax') if kw in _path_str), 'Ch 0')
        channel_names = [_ch0, 'paxillin', 'zyxin', 'actin']

    # Main channel = paxillin (shown on the large canvas)
    _MAIN_CH = next((i for i, n in enumerate(channel_names) if 'pax' in n.lower()), 0)

    # Old-format image fallback
    recon_images_dir = result_dir / 'recon' / 'images'
    old_img_files: list = []
    if images_raw is None and img_meta is None and result_dir != Path(''):
        old_img_files = sorted(recon_images_dir.glob('raw_*.tif'))

    pg_col   = 'patch_group' if 'patch_group' in df.columns else 'group'
    cond_col = 'condition_name' if 'condition_name' in df.columns else 'condition'

    grp_to_cond: dict = {}
    for _, row in df[[pg_col, cond_col]].dropna().drop_duplicates().iterrows():
        grp_to_cond[str(row[pg_col])] = str(row[cond_col])

    if images_raw is not None and img_meta is not None:
        unique_groups = sorted(img_meta['group'].astype(str).unique())
        def _get_canvas(group_key: str) -> np.ndarray:
            matches = img_meta[img_meta['group'].astype(str) == group_key]
            if matches.empty:
                return np.zeros((512, 512), dtype=np.float32)
            arr = images_raw[int(matches.iloc[0]['frame'])].astype(np.float32)
            return _display_norm(arr)
    else:
        unique_groups = sorted(p.stem[4:] for p in old_img_files)
        def _get_canvas(group_key: str) -> np.ndarray:
            p = recon_images_dir / f'raw_{group_key}.tif'
            arr = tifffile.imread(str(p)).astype(np.float32)
            if arr.ndim == 3:
                arr = arr[0]
            return _display_norm(arr)

    img_options = {f"{grp_to_cond.get(g, '?')} | {g}": g
                   for g in unique_groups}

    # Normalize frame numbers: strip leading zeros so control_f0000 == control_f0
    import re as _frame_re
    def _norm_fkey(k: str) -> str:
        return _frame_re.sub(r'_f0*(\d)', r'_f\1', k)

    def _display_norm(arr: np.ndarray, pct: float = 99.9) -> np.ndarray:
        """Clip at percentile then scale to [0, 1] for display."""
        hi = float(np.percentile(arr, pct))
        if hi <= 0:
            hi = float(arr.max())
        return np.clip(arr, 0, hi) / hi if hi > 0 else arr

    images_allch_norm: dict = {_norm_fkey(k): v for k, v in images_allch.items()}

    # ── Label storage ─────────────────────────────────────────────────────────
    labels: dict[str, str] = {}   # filename → label
    _state: dict = {}

    # ── Bokeh figure ──────────────────────────────────────────────────────────
    init_group = unique_groups[0]
    init_arr   = _get_canvas(init_group)
    H, W = init_arr.shape[:2]

    img_src = ColumnDataSource(dict(
        image=[np.ascontiguousarray(np.flipud(init_arr))],
        x=[0], y=[0], dw=[W], dh=[H],
    ))
    rects_src = ColumnDataSource(dict(
        x=[], y=[], width=[], height=[], fill_color=[], fill_alpha=[],
        line_color=[], df_idx=[],
    ))
    sel_src = ColumnDataSource(dict(x=[], y=[], width=[], height=[]))

    # ── Per-channel patch display ─────────────────────────────────────────────
    _n_canvas_ch = (next(iter(images_allch.values())).shape[0]
                    if images_allch else 0)
    _n_ch = (patches_allch.shape[1] if patches_allch is not None
             else _n_canvas_ch)   # fall back to channel count from full-canvas images
    _ps   = patches_allch.shape[2] if patches_allch is not None else 32
    _blank = np.zeros((_ps, _ps), dtype=np.float32)

    ch_srcs, ch_figs = [], []
    for _ci in range(_n_ch):
        _src = ColumnDataSource(dict(
            image=[np.ascontiguousarray(np.flipud(_blank))],
            x=[0], y=[0], dw=[_ps], dh=[_ps],
        ))
        _fig = figure(
            width=225, height=250,
            x_range=Range1d(0, _ps), y_range=Range1d(0, _ps),
            title=(channel_names[_ci] if channel_names and _ci < len(channel_names) else f'Ch {_ci}'),
            tools='', toolbar_location=None,
        )
        _fig.image(
            image='image', source=_src,
            x=0, y=0, dw=_ps, dh=_ps,
            color_mapper=LinearColorMapper(palette=GRAY256, low=0.0, high=1.0),
        )
        ch_srcs.append(_src)
        ch_figs.append(_fig)

    canvas_fig = figure(
        width=720, height=720,
        x_range=Range1d(0, W), y_range=Range1d(0, H),
        title=(channel_names[_MAIN_CH]
               if channel_names and _MAIN_CH < len(channel_names)
               else 'main canvas'),
        tools='pan,wheel_zoom,reset,tap',
        toolbar_location='above',
    )
    gray_mapper = LinearColorMapper(palette=GRAY256, low=0.0, high=1.0)
    canvas_fig.image(
        image='image', source=img_src,
        x=0, y=0, dw=W, dh=H,
        color_mapper=gray_mapper,
    )
    canvas_fig.rect(
        'x', 'y', 'width', 'height', source=rects_src,
        fill_color='fill_color', fill_alpha='fill_alpha',
        line_color='line_color', line_width=1.5, line_alpha=1.0,
        nonselection_fill_color='fill_color', nonselection_fill_alpha='fill_alpha',
        nonselection_line_color='line_color', nonselection_line_alpha=1.0,
    )
    canvas_fig.rect(
        'x', 'y', 'width', 'height', source=sel_src,
        fill_alpha=0, line_color='white', line_width=2.8,
    )

    # ── Full-canvas channel views (read-only, linked ranges) ─────────────────
    # Exclude the main channel (paxillin) — it's already shown in the main canvas.
    _side_ch_indices = [ci for ci in range(_n_canvas_ch) if ci != _MAIN_CH]
    ch_canvas_srcs: list = []
    ch_canvas_figs: list = []
    _blank_canvas = np.zeros((H, W), dtype=np.float32)
    for _ci in _side_ch_indices:
        _src = ColumnDataSource(dict(
            image=[np.ascontiguousarray(np.flipud(_blank_canvas))],
            x=[0], y=[0], dw=[W], dh=[H],
        ))
        _fig = figure(
            width=400, height=400,
            x_range=canvas_fig.x_range, y_range=canvas_fig.y_range,
            title=(channel_names[_ci] if channel_names and _ci < len(channel_names) else f'Ch {_ci}'),
            tools='', toolbar_location=None,
        )
        _fig.image(
            image='image', source=_src,
            x='x', y='y', dw='dw', dh='dh',
            color_mapper=LinearColorMapper(palette=GRAY256, low=0.0, high=1.0),
        )
        _fig.rect(
            'x', 'y', 'width', 'height', source=rects_src,
            fill_alpha=0, line_color='line_color', line_width=0.6, line_alpha=0.8,
            nonselection_fill_alpha=0, nonselection_line_color='line_color',
            nonselection_line_alpha=0.8,
        )
        _fig.rect(
            'x', 'y', 'width', 'height', source=sel_src,
            fill_alpha=0, line_color='white', line_width=1.5,
        )
        ch_canvas_srcs.append(_src)
        ch_canvas_figs.append(_fig)

    # ── Rect builder ─────────────────────────────────────────────────────────
    def _rects_for_group(group_key: str, img_H: int) -> dict:
        mask = df[pg_col].astype(str) == group_key
        sub  = df[mask]
        xs, ys, ws, hs = [], [], [], []
        fills, alphas, lines, idxs = [], [], [], []
        for i, row in sub.iterrows():
            cx = row.get('canvas_cx', np.nan)
            cy = row.get('canvas_cy', np.nan)
            ps = int(row.get('ps', 32))
            if pd.isna(cx) or pd.isna(cy):
                continue
            fname = str(row.get('filename', ''))
            lbl   = labels.get(fname, '')
            color = FA_COLOR_MAP.get(lbl, UNLABELED_COLOR)
            xs.append(float(cx) * image_scale)
            ys.append((img_H - float(cy)) * image_scale)
            ws.append(float(ps) * image_scale)
            hs.append(float(ps) * image_scale)
            fills.append(color)
            alphas.append(0.7 if lbl else 0.1)
            lines.append(color)
            idxs.append(i)
        return dict(x=xs, y=ys, width=ws, height=hs,
                    fill_color=fills, fill_alpha=alphas,
                    line_color=lines, df_idx=idxs)

    _state.update(group=init_group, H=H, W=W)
    rects_src.data = _rects_for_group(init_group, H)

    def _update_ch_canvas(group_key: str, img_H: int, img_W: int) -> None:
        _key = _norm_fkey(group_key)
        if images_allch_norm and _key in images_allch_norm:
            ch_arr_allch = images_allch_norm[_key]   # (C, H', W')
            for _csrc, _ci in zip(ch_canvas_srcs, _side_ch_indices):
                if _ci < ch_arr_allch.shape[0]:
                    _cimg = _display_norm(ch_arr_allch[_ci].astype(np.float32))
                    _csrc.data = dict(
                        image=[np.ascontiguousarray(np.flipud(_cimg))],
                        x=[0], y=[0], dw=[_cimg.shape[1]], dh=[_cimg.shape[0]],
                    )
        elif images_allch_norm and ch_canvas_srcs:
            import sys as _sys
            print(f"[label] WARN: group_key {_key!r} not in images/allch "
                  f"(available: {sorted(images_allch_norm.keys())[:4]})", file=_sys.stderr)
            _blank = np.zeros((img_H, img_W), dtype=np.float32)
            for _csrc in ch_canvas_srcs:
                _csrc.data = dict(
                    image=[np.ascontiguousarray(np.flipud(_blank))],
                    x=[0], y=[0], dw=[img_W], dh=[img_H],
                )
        elif ch_canvas_srcs:
            _blank = np.zeros((img_H, img_W), dtype=np.float32)
            for _csrc in ch_canvas_srcs:
                _csrc.data = dict(
                    image=[np.ascontiguousarray(np.flipud(_blank))],
                    x=[0], y=[0], dw=[img_W], dh=[img_H],
                )

    _update_ch_canvas(init_group, H, W)

    def _load_group(group_key: str) -> None:
        arr    = _get_canvas(group_key)
        Hn, Wn = arr.shape[:2]
        img_src.data = dict(
            image=[np.ascontiguousarray(np.flipud(arr))],
            x=[0], y=[0], dw=[Wn], dh=[Hn],
        )
        canvas_fig.x_range.start, canvas_fig.x_range.end = 0, Wn
        canvas_fig.y_range.start, canvas_fig.y_range.end = 0, Hn
        rects_src.data = _rects_for_group(group_key, Hn)
        sel_src.data   = dict(x=[], y=[], width=[], height=[])
        _state.update(group=group_key, H=Hn, W=Wn)
        _update_ch_canvas(group_key, Hn, Wn)
        for _csrc in ch_srcs:
            _csrc.data = dict(
                image=[np.ascontiguousarray(np.flipud(_blank))],
                x=[0], y=[0], dw=[_ps], dh=[_ps],
            )

    # ── Widgets ───────────────────────────────────────────────────────────────
    img_selector = pn.widgets.Select(
        name='Image', options=img_options, value=init_group, width=440,
    )
    img_selector.param.watch(lambda e: _load_group(e.new), 'value')

    label_group = pn.widgets.RadioButtonGroup(
        name='Active label',
        options=LABEL_OPTIONS,
        value=LABEL_OPTIONS[0],
        button_type='default',
        width=560,
    )

    name_input = pn.widgets.TextInput(
        placeholder='Your name (used in save filename)…',
        width=240,
    )

    status_md = pn.pane.HTML(
        '<i style="color:#888;">Enter your name, select a label, then click a patch.</i>',
        width=560,
    )
    count_md = pn.pane.Markdown('**Labeled:** 0', width=120)

    def _update_count() -> None:
        count_md.object = f'**Labeled:** {len(labels)}'

    # ── Canvas tap handler ────────────────────────────────────────────────────
    def _on_tap(event: Tap) -> None:
        H_cur = _state['H']
        tap_cx = event.x / image_scale
        tap_cy = (H_cur - event.y) / image_scale

        xs_bk  = np.array(rects_src.data['x'],      dtype=float)
        ys_bk  = np.array(rects_src.data['y'],      dtype=float)
        df_idx = np.array(rects_src.data['df_idx'], dtype=int)
        if len(xs_bk) == 0:
            return

        cx_arr = xs_bk / image_scale
        cy_arr = (H_cur - ys_bk) / image_scale
        dists  = np.sqrt((cx_arr - tap_cx)**2 + (cy_arr - tap_cy)**2)
        near_i  = int(np.argmin(dists))
        near_df = int(df_idx[near_i])

        # Ignore clicks outside the nearest patch boundary
        ws_bk = np.array(rects_src.data['width'],  dtype=float)
        hs_bk = np.array(rects_src.data['height'], dtype=float)
        hw = ws_bk[near_i] / image_scale / 2
        hh = hs_bk[near_i] / image_scale / 2
        if abs(tap_cx - cx_arr[near_i]) > hw or abs(tap_cy - cy_arr[near_i]) > hh:
            return

        # Save position before refresh (indices shift after update)
        sel_x = float(xs_bk[near_i])
        sel_y = float(ys_bk[near_i])
        sel_w = float(rects_src.data['width'][near_i])
        sel_h = float(rects_src.data['height'][near_i])

        row    = df.iloc[near_df]
        fname  = str(row.get('filename', ''))
        active = label_group.value
        labels[fname] = active

        # Refresh patch colours
        rects_src.data = _rects_for_group(_state['group'], _state['H'])
        sel_src.data   = dict(x=[sel_x], y=[sel_y], width=[sel_w], height=[sel_h])

        color = FA_COLOR_MAP.get(active, '#ffffff')
        status_md.object = (
            f'<span style="font-size:13px;">'
            f'Labeled <b>{Path(fname).stem}</b> → '
            f'<span style="color:{color};font-weight:bold;">{active}</span>'
            f'</span>'
        )
        # Update per-channel patch display
        if patches_allch is not None and near_df < len(patches_allch):
            allch = patches_allch[near_df]   # (C, ps, ps) float32
            for _ci, _csrc in enumerate(ch_srcs):
                if _ci < allch.shape[0]:
                    _arr = allch[_ci].astype(np.float32)
                    _csrc.data = dict(
                        image=[np.ascontiguousarray(np.flipud(_arr))],
                        x=[0], y=[0], dw=[_arr.shape[1]], dh=[_arr.shape[0]],
                    )
        elif ch_srcs and images_allch_norm:
            # Extract patch region on the fly from full-canvas allch images
            _grp_key = _norm_fkey(_state['group'])
            _allch_c = images_allch_norm.get(_grp_key)
            if _allch_c is not None:
                _cx   = int(float(row.get('canvas_cx', 0)))
                _cy   = int(float(row.get('canvas_cy', 0)))
                _ps_r = int(row.get('ps', _ps))
                _half = _ps_r // 2
                _C, _H_c, _W_c = _allch_c.shape
                _y0 = max(0, _cy - _half); _y1 = min(_H_c, _cy + _half)
                _x0 = max(0, _cx - _half); _x1 = min(_W_c, _cx + _half)
                for _ci, _csrc in enumerate(ch_srcs):
                    if _ci < _C:
                        _patch = _display_norm(_allch_c[_ci, _y0:_y1, _x0:_x1].astype(np.float32))
                        _csrc.data = dict(
                            image=[np.ascontiguousarray(np.flipud(_patch))],
                            x=[0], y=[0], dw=[_patch.shape[1]], dh=[_patch.shape[0]],
                        )
        _update_count()

    canvas_fig.on_event(Tap, _on_tap)

    # ── Double-click to remove label ──────────────────────────────────────────
    def _on_doubletap(event: DoubleTap) -> None:
        H_cur  = _state['H']
        tap_cx = event.x / image_scale
        tap_cy = (H_cur - event.y) / image_scale

        xs_bk  = np.array(rects_src.data['x'],      dtype=float)
        ys_bk  = np.array(rects_src.data['y'],      dtype=float)
        df_idx = np.array(rects_src.data['df_idx'], dtype=int)
        if len(xs_bk) == 0:
            return

        cx_arr = xs_bk / image_scale
        cy_arr = (H_cur - ys_bk) / image_scale
        dists  = np.sqrt((cx_arr - tap_cx)**2 + (cy_arr - tap_cy)**2)
        near_i = int(np.argmin(dists))

        ws_bk = np.array(rects_src.data['width'],  dtype=float)
        hs_bk = np.array(rects_src.data['height'], dtype=float)
        hw = ws_bk[near_i] / image_scale / 2
        hh = hs_bk[near_i] / image_scale / 2
        if abs(tap_cx - cx_arr[near_i]) > hw or abs(tap_cy - cy_arr[near_i]) > hh:
            return

        row   = df.iloc[int(df_idx[near_i])]
        fname = str(row.get('filename', ''))
        if fname in labels:
            del labels[fname]
            rects_src.data = _rects_for_group(_state['group'], _state['H'])
            sel_src.data   = dict(x=[], y=[], width=[], height=[])
            status_md.object = (
                f'<span style="font-size:13px;">Removed label from '
                f'<b>{Path(fname).stem}</b></span>'
            )
            _update_count()

    canvas_fig.on_event(DoubleTap, _on_doubletap)

    # ── Finish & Save ─────────────────────────────────────────────────────────
    finish_btn = pn.widgets.Button(
        name='Finish & Save', button_type='success', width=160,
    )

    def _on_finish(event) -> None:
        if not labels:
            status_md.object = '<i style="color:#e55;">No labels to save.</i>'
            return
        annotator = name_input.value.strip().replace(' ', '_') or 'unknown'
        stamp = datetime.now().strftime('%Y%m%d_%H%M')
        h5_stem = Path(h5_path).stem
        out = Path(h5_path).parent / f'{h5_stem}_{annotator}_{stamp}.csv'
        out.parent.mkdir(parents=True, exist_ok=True)
        rows = [{'filename': fn, 'label': lbl, 'annotator': annotator}
                for fn, lbl in labels.items()]
        pd.DataFrame(rows).to_csv(out, index=False)
        status_md.object = (
            f'<span style="color:#3c3;font-size:13px;font-weight:bold;">'
            f'✓ Saved {len(labels)} labels → {out.name}</span>'
        )

    finish_btn.on_click(_on_finish)

    # ── Resume from previous CSV ──────────────────────────────────────────────
    _h5_dir = Path(h5_path).parent
    _h5_stem = Path(h5_path).stem
    _prev_csvs = sorted(_h5_dir.glob(f'{_h5_stem}_*.csv'))
    _resume_default = str(_prev_csvs[-1]) if _prev_csvs else ''
    resume_input = pn.widgets.TextInput(
        value=_resume_default,
        placeholder=f'Path to previous labels CSV (default folder: {_h5_dir})',
        width=500,
    )
    resume_btn = pn.widgets.Button(name='Load CSV', button_type='primary', width=100)

    def _on_resume(event) -> None:
        csv_path = resume_input.value.strip()
        if not csv_path:
            status_md.object = '<i style="color:#e55;">Paste a CSV path first.</i>'
            return
        p = Path(csv_path)
        if not p.exists():
            status_md.object = f'<i style="color:#e55;">File not found: {p}</i>'
            return
        try:
            prev = pd.read_csv(p)
            if 'filename' not in prev.columns or 'label' not in prev.columns:
                status_md.object = '<i style="color:#e55;">CSV must have filename and label columns.</i>'
                return
            loaded = 0
            for _, row in prev.dropna(subset=['filename', 'label']).iterrows():
                fname = str(row['filename'])
                lbl   = str(row['label']).strip()
                if lbl and lbl in LABEL_OPTIONS:
                    labels[fname] = lbl
                    loaded += 1
            rects_src.data = _rects_for_group(_state['group'], _state['H'])
            _update_count()
            status_md.object = (
                f'<span style="color:#38a;font-size:13px;font-weight:bold;">'
                f'✓ Resumed {loaded} labels from {p.name}</span>'
            )
        except Exception as e:
            status_md.object = f'<i style="color:#e55;">Error loading CSV: {e}</i>'

    resume_btn.on_click(_on_resume)

    # ── Layout ────────────────────────────────────────────────────────────────
    toolbar = pn.Row(
        pn.pane.HTML('<b style="line-height:2.2;">Active label:</b>', width=100),
        label_group,
        pn.Spacer(width=20),
        count_md,
        pn.Spacer(width=20),
        finish_btn,
    )
    resume_row = pn.Row(
        pn.pane.HTML('<b style="line-height:2.2;">Resume:</b>', width=80),
        resume_input,
        pn.Spacer(width=10),
        resume_btn,
    )

    # ── Right panel: 3 other-channel full canvases + 4 patch thumbnails ─────────
    _side_canvas_row  = (pn.Row(*[pn.pane.Bokeh(f) for f in ch_canvas_figs])
                         if ch_canvas_figs else None)
    _patch_thumb_row  = (pn.Row(*[pn.pane.Bokeh(f) for f in ch_figs])
                         if ch_figs else None)

    _right_children = []
    if _side_canvas_row is not None:
        _right_children.append(pn.pane.HTML('<b>Full canvas — other channels</b>'))
        _right_children.append(_side_canvas_row)
    if _patch_thumb_row is not None:
        _right_children.append(pn.pane.HTML('<b>Selected patch — all channels</b>'))
        _right_children.append(_patch_thumb_row)

    if _right_children:
        right_panel = pn.Column(*_right_children)
        main_row = pn.Row(pn.pane.Bokeh(canvas_fig), pn.Spacer(width=100), right_panel)
    else:
        main_row = pn.pane.Bokeh(canvas_fig)

    return pn.Column(
        pn.pane.HTML(
            f'<h2>Patch Labeling Tool &nbsp;·&nbsp; {Path(h5_path).name}'
            + (f' &nbsp;<span style="font-size:14px;font-weight:normal;color:#888;">in {location}</span>' if location else '')
            + '</h2>',
            sizing_mode='stretch_width',
        ),
        pn.Row(
            pn.pane.HTML('<b style="line-height:2.2;">Annotator:</b>', width=80),
            name_input,
            pn.Spacer(width=20),
            img_selector,
        ),
        resume_row,
        toolbar,
        status_md,
        main_row,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('h5', nargs='+', help='One or more interactive.h5 files')
    p.add_argument('--port', type=int, default=5007)
    p.add_argument('--serve', action='store_true',
                   help='Bind to 0.0.0.0 for network access (lab server mode). '
                        'Lab members open http://<server-ip>:<port>/<name> in their browser.')
    p.add_argument('--nas-mount', default=None,
                   help='NAS mount prefix to strip from paths, e.g. /mnt/p/')
    p.add_argument('--nas-name', default=None,
                   help='Human-readable NAS label, e.g. "GardelNas Expansion"')
    return p.parse_args()


def _get_h5_path() -> str:
    sess = pn.state.session_args
    if 'args' in sess and sess['args']:
        arg = sess['args'][0]
        return arg.decode() if isinstance(arg, bytes) else str(arg)
    if len(sys.argv) > 1:
        return sys.argv[1]
    print('Usage: python scripts/label_patches.py path/to/interactive.h5',
          file=sys.stderr)
    sys.exit(1)


if pn.state.served:
    _h5 = _get_h5_path()
    build_labeler(_h5).servable()

if __name__ == '__main__':
    import socket
    args = _parse_args()
    h5_paths = [str(Path(p).resolve()) for p in args.h5]

    def _location(h: str) -> str:
        """Build the 'in <location>' subtitle for a given H5 path."""
        p = Path(h).parent   # folder containing interactive.h5
        if args.nas_mount and args.nas_name:
            mount = args.nas_mount.rstrip('/')
            rel   = str(p).removeprefix(mount).lstrip('/')
            return f'{args.nas_name}: {rel}'
        return ''

    # Build route dict: one H5 → serve at '/', multiple → serve at '/<stem>'
    if len(h5_paths) == 1:
        routes = {'/': lambda h=h5_paths[0], loc=_location(h5_paths[0]): build_labeler(h, loc)}
    else:
        routes = {f'/{Path(h).stem}': (lambda h=h, loc=_location(h): build_labeler(h, loc))
                  for h in h5_paths}

    if args.serve:
        try:
            _s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            _s.connect(('8.8.8.8', 80))
            host_ip = _s.getsockname()[0]
            _s.close()
        except Exception:
            host_ip = '0.0.0.0'
        print(f'[label] Serving in lab mode on port {args.port}')
        for route, h in zip(routes.keys(), h5_paths):
            print(f'[label]   http://{host_ip}:{args.port}{route}  →  {h}')
        pn.serve(routes,
                 address='0.0.0.0',
                 port=args.port,
                 allow_websocket_origin=['*'],
                 show=False,
                 autoreload=False)
    else:
        for route, h in zip(routes.keys(), h5_paths):
            print(f'[label]   http://localhost:{args.port}{route}  →  {h}')
        pn.serve(routes,
                 show=True, port=args.port, autoreload=False)
