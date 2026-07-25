# Context: Data Viewer (`scripts/data_viewer.py`)

This document gives Claude Code enough background to continue development of the
standalone data viewer without re-explaining prior decisions.

For the model+UMAP interactive viewer see `context_interactive_viewer.md`.

---

## What the viewer is

A Panel + Bokeh web app for exploring raw multi-channel FA patch data — no model
outputs.  Multiple datasets are served at separate URL routes from one process.

Layout per dataset:
```
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  Dataset N: ds_name — pax-ch1 · vinc-ch0 · zyxin-ch2 · actin-ch3       │
  ├─────────────────────────────┬────────────────────────────────────────────┤
  │  Paxillin canvas (520px)    │  Side channel canvases (265px each, linked)│
  │  + pixel overlay checkboxes │  + grid rect overlay only (no px overlay)  │
  │  + grid colour checkboxes   │                                            │
  ├─────────────────────────────┴────────────────────────────────────────────┤
  │  [Histogram pax-ch1]   [Detail text]   [Patch thumbnails — all channels] │
  └──────────────────────────────────────────────────────────────────────────┘
```

---

## How to run (current server: 128.135.108.109, port 5008)

```bash
# Four datasets at /vinc  /ppax  /pfak  /nih3t3
nohup python scripts/data_viewer.py \
    /mnt/p/image_service/data/FA_patch_data/cio/vinc/data.h5 \
    /mnt/p/image_service/data/FA_patch_data/cio/ppax/data.h5 \
    /mnt/p/image_service/data/FA_patch_data/cio/pfak/data.h5 \
    /mnt/p/image_service/data/FA_patch_data/cio/nih3t3/data.h5 \
    --port 5008 --serve > /tmp/data_viewer.log 2>&1 &
```

`--serve` binds to `0.0.0.0`.  Each dataset gets a route `/{ds_name}`.
With one path, the app is served at `/`.

---

## H5 format (`data.h5`)

Built by `scripts/pack_data_h5.py`.

```
patches/raw       float32 (N, 32, 32)   all patches, all conditions
images/raw        float32 (M, H, W)     paxillin frames (channel 0 in ch_keys)
images/{ch}       float32 (M, H, W)     extra channels (same M frame order)
images/meta       bytes (CSV)           group, frame (row index), condition_name, frame_idx
meta/csv          bytes (CSV)           filename, condition_name, group, frame_idx,
                                        canvas_cx, canvas_cy, ps,
                                        mean_intensity, annotation_label, annotation_label_name
attrs: pad_size, image_scale, dataset, n_patches, n_frames, channels (JSON)
```

`channels` attr: JSON list e.g. `["pax","vinc","zyx","act"]`.
`images/raw` = first entry (pax). `images/{k}` = remaining entries.

---

## Channel label convention

```python
_CH_IDX   = {'pax': 1, 'zyx': 2, 'act': 3, 'vinc': 0, 'ppax': 0, 'pfak': 0}
_CH_SHORT  = {'pax': 'pax', 'zyx': 'zyxin', 'act': 'actin', 'vinc': 'vinc', …}

# Canvas title labels: "pax-ch1", "vinc-ch0", "zyxin-ch2", "actin-ch3"
ch_canvas_labels = [f'{_CH_SHORT.get(k,k)}-ch{_CH_IDX.get(k,"?")}' for k in ch_keys]

# Dataset header: "Dataset 1: vinc — pax-ch1 · vinc-ch0 · zyxin-ch2 · actin-ch3"
```

The `is_pax` check in `_all_channel_figure` uses **index position** (`i == 0`),
NOT name substring — important because 'pax' is a substring of 'ppax'.

---

## Canvas and image loading

`_read_h5` returns:
```
df, patches_raw, images_raw, img_meta,
images_allch,      # {ch_key: float32 (M, H, W)} — extra channels only
channel_names,     # display labels list
ch_keys,           # raw key list e.g. ['pax','vinc','zyx','act']
pad_size, image_scale
```

Per-frame channel data is stored as a flat `images_allch[ch_key]` array indexed
by the same `frame` row from `images/meta` — NOT a per-frame dict.

`_all_channel_figure(i, name, arr, H, W, ...)` builds one Bokeh figure per
channel. For `i == 0` (pax): pixel overlay RGBA renderer is added.
Side channels: pixel overlay is skipped; only grid rects are drawn.

---

## Pixel intensity overlay (paxillin only)

Four thresholds (per-pixel on float32 CIO values):
| Threshold | Colour |
|-----------|--------|
| < 0 | blue `#4488FF` |
| > 1 | dark green `#009944` |
| > 2 | magenta `#FF44FF` |
| > 4 | red `#FF4444` |

`_on_pixel_overlay(e)`: rebuilds a uint32 RGBA array from the current pax frame,
pushes to `overlay_src` (`image_rgba` renderer on pax canvas only).
Called when any pixel-highlight checkbox changes or when the group changes.

Checkboxes: `ck_blue`, `ck_green`, `ck_magenta`, `ck_red` — Panel `Checkbox`.

---

## Grid rect overlay

Drawn on ALL channel canvases (pax + side). Three border colour bands based on
**paxillin patch max intensity** (`patch_maxes`):
| patch max | Border |
|-----------|--------|
| ≤ 2 | invisible (lw=0) |
| > 2 | magenta, lw=0.5 |
| > 4 | red, lw=1.5 |

Default: all grid checkboxes off.  Label: "based on pax patch intensity max".

`rects_src` is a single `ColumnDataSource` shared by all channel canvas figures
(they all render the same rects, since canvases are aligned).

---

## Dim toggle

`dim_toggle` Panel Toggle (Dim vmax=2).  On toggle:
```python
gray_mapper.high = 2.0 if e.new else 1.0
for _m in side_mappers:   # list of LinearColorMapper for each side channel
    _m.high = 2.0 if e.new else 1.0
```
All channel canvases dim together. Button placed in bottom row beside detail text.

---

## Aspect ratio sizing

Canvas figures are sized proportionally to the image H/W:
```python
_aspect = H / W
_pax_w, _pax_h   = 520, int(520 * _aspect)
_side_w, _side_h = 265, int(265 * _aspect)
```
Computed once per `build_app` call from the initial frame dimensions.

---

## Layout skeleton

```python
info_bar   = pn.Row(dataset_info_html, HSpacer())
pax_col    = pn.Column(pn.pane.Bokeh(pax_fig), width=_pax_w + 10)
right_col  = pn.Column(img_select, overlay_row, grid_row, side_row)
canvas_row = pn.Row(pax_col, right_col)
detail_row = pn.Row(pn.Column(detail_info, width=465), dim_toggle)
bottom_row = pn.Row(hist_col, pn.Column(detail_row, patch_pane))
return pn.Column(info_bar, canvas_row, bottom_row)
```

`overlay_row` = pixel highlight checkboxes (< 0, > 1, > 2, > 4).
`grid_row` = grid colour checkboxes with "based on pax patch intensity max" label.
`side_row` = side-channel Bokeh canvases with linked ranges.

Widget widths `_W0, _W1, _W2, _W3 = 105, 140, 115, 85` are tuned to align
pixel-highlight checkboxes and grid checkboxes into matching columns horizontally.

---

## Histogram

- Paxillin pixel intensity distribution (all patches flattened).
- Title: `f'pax-ch1  N={n}\n<0:{n_b}  >1:{n_g}  >2:{n_m}  >4:{n_r}'`
  (dataset name NOT shown — histogram is always pax regardless of dataset).
- Size: `figsize=(3.0, 3.0)`, `dpi=100`.  Container: `hist_col width=320`.

---

## Patch thumbnails (bottom right)

On canvas click → shows `n_ch` patches (one per channel) in a row.
```python
figsize = (2.5 * n_ch, 2.5)
patch_pane width = int(2.5 * 100 * n_ch)
```

`detail_info` is a single `white-space:nowrap` HTML line — no wrapping, no
fixed width, to avoid pushing the patch pane sideways.

---

## Multi-dataset routing

```python
routes = {}
for i, path in enumerate(h5_paths):
    ds = Path(path).parent.name   # e.g. 'vinc'
    routes[f'/{ds}'] = (lambda h=path, idx=i+1: build_app(h, ds_idx=idx))
pn.serve(routes, address='0.0.0.0', port=args.port, ...)
```

`build_app(data_h5, ds_idx=1)` — `ds_idx` is used only for the dataset header.
Single-path mode: `pn.serve(build_app(args.h5[0]), ...)` (served at `/`).

---

## Known issue: 5× scale mismatch between canvas and patches

`patches/raw` (patchprep pipeline) and `images/raw` (frameextract pipeline)
were generated with different `normalize_cell_insideoutside` scale values:
- `patchprep`: uses default `scale=5.0` (never overridden in cio configs)
- `frameextract`: uses `scale=1.0` (explicitly set in cio configs)

Net effect: `patches_raw[i]` ≈ `canvas_crop / 5`.

**Viewer-side workaround** (still in place):
- Canvas displayed with `/10` divisor; patches displayed with `vmax` tuned accordingly.
- Pixel-overlay thresholds applied to canvas values (not patch values).

**Real fix** (not yet done): realign patchprep + frameextract configs to use the
same scale, then re-pack `data.h5`. Remove viewer-side divisors after repacking.

---

## Things to know / traps

- **`is_pax = i == 0`** (index check), never `'pax' in name.lower()` — the latter
  matches 'ppax' and would apply pax overlay to the wrong channel.
- **`side_mappers` list** accumulates one `LinearColorMapper` per side channel;
  the dim toggle iterates this list. Don't forget to append to it when adding
  a new channel canvas.
- **Grid checkboxes default to off** — `value=False` in all three `pn.Checkbox`.
- **`pn.layout.HSpacer()`** at the end of `grid_row` prevents
  `sizing_mode='stretch_width'` from spreading the checkboxes apart.
- **`_NORM_FKEY = re.compile(r'_f0*(\d)')`** strips frame-number prefix when
  grouping patches — used in title/label generation, not routing.
