# Context: Interactive Model Viewer — `view_interactive.py`

This document gives Claude Code enough background to continue development of the
interactive model viewer system without re-explaining prior decisions.

---

## What this system is

A Panel + Bokeh web app for exploring AE model outputs (latents, UMAP, predictions,
reconstructions) alongside the raw multi-channel microscopy data.

Two directions of exploration:
- **UMAP → Canvas**: tap a UMAP dot → canvas auto-pans to the patch, detail panel
  updates, side-channel patch thumbnails appear.
- **Canvas → UMAP**: click a patch rectangle on the canvas → UMAP dot highlighted
  with the patch's FA-label colour, detail panel updates.

---

## Files in this system

| Script | Role |
|--------|------|
| `scripts/view_interactive.py` | Panel + Bokeh viewer (this doc) |
| `scripts/pack_model_h5.py` | Pack one model result dir → `model.h5` |
| `scripts/pack_data_h5.py` | Pack patches + images → `data.h5` (shared across models) |

---

## H5 two-file design

### `data.h5` — static, shared across models (built by `pack_data_h5.py`)

```
patches/raw       float32 (N, 32, 32)   all patches, all conditions
images/raw        float32 (M, H, W)     paxillin frames
images/{ch}       float32 (M, H, W)     extra channels: vinc / zyx / act / ppax / pfak
images/meta       bytes (CSV)           group, frame (row index in images/raw),
                                        condition_name, frame_idx
meta/csv          bytes (CSV)           filename, condition_name, group, frame_idx,
                                        canvas_cx, canvas_cy, ps,
                                        mean_intensity, annotation_label, annotation_label_name
attrs: pad_size, image_scale, dataset, n_patches, n_frames, channels (JSON list)
```

`channels` attr is a JSON list like `["pax","vinc","zyx","act"]`.
`images/raw` = first channel (pax). `images/{ch}` = remaining channels (same frame order).

### `model.h5` — per-model outputs (built by `pack_model_h5.py`)

```
meta/csv          bytes (CSV)           filename, z_*, UMAP_1, UMAP_2,
                                        fa_pred, fa_prob_*, pos_pred, pos_prob_*,
                                        annotation_label, split, …
patches/recon     float32 (N, 32, 32)   AE reconstructions
plots/{name}      bytes (PNG)           analysis plots (MSE etc.)
attrs: pad_size, image_scale, result_dir, n_patches, model_name
```

---

## `pack_model_h5.py` — what it handles

Primary CSV candidates (tried in order, first found wins):
1. `{result_dir}/analysis/analysis_results.csv`
2. `{result_dir}/latents_newdata.csv`
3. `{result_dir}/latents.csv`

UMAP (tried in order):
1. `UMAP_1`/`UMAP_2` already in primary CSV → use as-is
2. Merge from `analysis/analysis_results.csv` if different from primary
3. **Load a saved UMAP model** (`umap_all_model.pkl`) from any of:
   `fa_cls_lat8/`, `fa_cls_zproj/`, `fa_cls_zrecon/`, `vis_lat8/`,
   `vis_lat8dist8/`, `analysis/` — transform `z_*` latents with `joblib.load` +
   `umap_model.transform(z_arr)`. Needed when only UMAP PNG plots are saved,
   not coordinates.

FA predictions (first found): `fa_cls_lat8/` → `fa_cls_zproj/` → `fa_cls_zrecon/`
 — files: `predictions_all.csv` or `classification_results.csv`.
 Handles both `proba_*` and `prob_*` column prefixes.

Position predictions: same pattern with `pos_cls_*` subdirs.

Recon patches — two formats handled:
- **New (stacked)**: `recon/patches_recon.tif` + `recon/patches_index.csv`
- **Old (individual TIFFs)**: `recon/patches/recon_{split}_{stem}.tif`
  (or `recon_{stem}.tif` without split prefix). Split comes from `split`
  column in primary CSV.

---

## `load_sources` — data merge logic

Two-file mode (`data.h5` + `model.h5`):
- Images + raw patches come from `data.h5` (higher quality, all conditions)
- Model df is left-joined with `data.h5` `meta/csv` on `filename` to get
  `canvas_cx`, `canvas_cy`, `ps`, `mean_intensity`, `annotation_label`
- `patches_raw` is re-indexed from `data.h5` row order to model df row order;
  unmatched model patches get zero arrays (warning printed, no crash)
- Extra channel images loaded from `data.h5`; `ch_keys` from `channels` attr

Mismatched patches (model has patches not in data.h5, or vice versa):
- Model-only patches: NaN `canvas_cx`/`canvas_cy` → skipped in rect drawing
  and side-patch crops; no crash
- Data-only patches: don't appear in UMAP at all

---

## Viewer layout

```
pn.Column(
  header,
  pn.Row(
    left_canvas_col,          # fixed structure — extra channels nested here
    Spacer(12, sizing_mode='fixed'),
    detail_col,               # dynamic content — grows on click
    Spacer(8, sizing_mode='fixed'),
    side_patch_col,           # 3 extra-channel patch crops, appears on click
  )
)

left_canvas_col = pn.Column(
  pn.Row(left_col, Spacer(12), canvas_col),   # UMAP | pax canvas
  pn.Row(*extra_ch_panes),                     # 3 linked channel canvases below
)
```

**Why nested like this**: `detail_col` and `side_patch_col` grow when a patch is
clicked. If `extra_ch_panes` were a sibling of `detail_col` in the outer Column,
they would shift down when `detail_col` grew. Nesting them inside `left_canvas_col`
(which has fixed content) isolates them from that height change.

**Spacers must use `sizing_mode='fixed'`** — the global
`pn.extension(sizing_mode='stretch_width')` would otherwise make them expand and
create a large gap between left_canvas_col and detail_col.

### left_col (non data-only mode)
- `color_select` Bokeh Select (FA type / Position) with a JS callback that
  swaps the `color` column in `umap_src.data` client-side
- `umap_filter` Panel RadioButtonGroup ("All patches" / "Labeled only")
  — "Labeled only" filters to `annotation_label >= 0` (human labels from
  `data.h5`, NOT predictions)
- UMAP Bokeh scatter (500×500px) with tap + hover tools

### canvas_col (pax canvas)
- Image selector dropdown (condition | group_key)
- Dim toggle: vmax=1.0 ↔ 2.0, updates `gray_mapper` + all `extra_ch_mappers`
- Bokeh canvas 520×520px with `_flip_for_bokeh` (flipud so row-0 at top)
- Patch rect overlay: coloured by max intensity (>4 red, >2 magenta, else hidden)
- White border rect on selected patch (`sel_src`) — also drawn on extra ch canvases

### extra_ch_panes (3 × 340px Bokeh figures)
- Share `x_range`/`y_range` with pax figure → synchronized pan/zoom
- `sel_src` rect renderer added to each → selection box synced automatically
- Updated in `_load_group` by reading `extra_ch_images[k][frame_row]`

### detail_col (right panel)
- Prediction text (FA type + position, colour-coded)
- L1 stat line: `L1: 0.1823 → p64 | p10=… p50=… p90=…` — computed at startup
  from `patch_l1s = |patches_raw|.mean(axis=(1,2))`
- Raw + Recon patch figure (matplotlib, 2.6×2.6 in each panel)
- Legend (all FA + position labels with colour swatches)
- MSE plot button (opens PNG in new window)

### side_patch_col (165px column)
- 3 matplotlib thumbnails (1.56×1.56 in each) — crops of extra channel frames
  at the selected patch's `canvas_cx`, `canvas_cy`, `ps` coordinates
- Updated in `_show_detail`; empty before first click

---

## Key coordinate conventions

```
canvas_cx, canvas_cy  = patch centre in unpadded canvas space (px)
                        patch covers [cy-ps//2 : cy+ps//2, cx-ps//2 : cx+ps//2]

Bokeh display (flipud):
  bx = canvas_cx * image_scale
  by = (H - canvas_cy) * image_scale
```

Auto-pan on UMAP click keeps equal x/y span (preserves aspect ratio):
```python
xe = min(W_sc, bx + margin)
xs = max(0,    xe - 2*margin)   # slide start back, don't clip the span
ye = min(H_sc, by + margin)
ys = max(0,    ye - 2*margin)
```

---

## Channel label format

Module-level dicts used in both data_viewer.py and view_interactive.py:
```python
_CH_IDX   = {'pax': 1, 'zyx': 2, 'act': 3, 'vinc': 0, 'ppax': 0, 'pfak': 0}
_CH_SHORT  = {'pax': 'pax', 'zyx': 'zyxin', 'act': 'actin', 'vinc': 'vinc', …}

def _ch_label(key): return f'{_CH_SHORT.get(key,key)}-ch{_CH_IDX.get(key,"?")}'
# → 'pax-ch1', 'vinc-ch0', 'zyxin-ch2', 'actin-ch3'
```

---

## Robustness — mismatched patches

Patches in model but not in data.h5:
- `canvas_cx`/`canvas_cy` are NaN after left join → skipped in rect drawing
- `_on_umap_tap`: `if not pd.isna(cx) and not pd.isna(cy)` guards canvas update
- `patches_raw[idx]` = zeros → blank patch image shown (no crash)

Groups in model but not in `img_meta`:
- `_unique_groups_set` checked before calling `_load_group`; warning printed if missing
- `_get_canvas` returns `None` if `_get_frame` finds no match; `_load_group` returns early

Frame index out of bounds for extra channels:
- `if frame_row >= len(frames): continue` (warning printed)

Empty patch crop (near-edge coordinates):
- `if patch.size == 0: continue` (warning printed)

---

## UMAP filter internals

```python
_umap_all_data     = {k: np.array(v) for k, v in umap_data.items()}
labeled_mask       = df['annotation_label'].fillna(-1).values >= 0
_umap_labeled_data = {k: v[labeled_mask] for k, v in _umap_all_data.items()}
# umap_data['idx'] holds original df row indices — preserved in filtered subset
```

`_on_umap_tap` uses `int(umap_src.data['idx'][new[0]])` (not `int(new[0])`)
so it always gets the original df row index regardless of which filter is active.

---

## Running the viewer (current server: 128.135.108.109)

Port layout:
| Port | Process |
|------|---------|
| 5006 | `view_interactive.py` — supcon model (vinc data.h5) |
| 5007 | Reserved for label_patches.py labeller (other computer) |
| 5008 | `data_viewer.py` — all 4 datasets |
| 5009 | `view_interactive.py` — semisup_both model (vinc data.h5) |

```bash
# Pack model.h5 from a result directory
python scripts/pack_model_h5.py /path/to/result_dir

# Launch model viewer (two-file mode)
nohup python scripts/view_interactive.py \
    /mnt/p/image_service/data/FA_patch_data/cio/vinc/data.h5 \
    /path/to/result_dir/model.h5 \
    --port 5009 --serve > /tmp/view_model.log 2>&1 &
```

---

## Things to know / traps

- **`pn.extension(sizing_mode='stretch_width')` is global** — any `pn.Spacer`
  without `sizing_mode='fixed'` will stretch and create unwanted gaps.
- **Extra channel canvases are in `left_canvas_col`**, not in the outer Column.
  Don't move them to a sibling row of `detail_col` — that causes the
  "shift on click" layout bug.
- **`_load_group` clears `highlight_src` with 3 keys**: `dict(x=[], y=[], color=[])`
  (not just `x`, `y`) — the UMAP highlight scatter uses a `color` column.
- **UMAP filter swaps entire `umap_src.data`** — the `idx` column in the
  filtered subset still holds original df row indices, so tap handlers work.
- **`dim_toggle` affects all mappers**: `gray_mapper` (pax) + every entry in
  `extra_ch_mappers` list.
- **`_refresh_detail` must be called after `dim_toggle` changes** so the
  currently-shown patch thumbnails re-render at the new vmax.
- **Recon patches are read from `model.h5 patches/recon`** at startup — no
  on-the-fly computation.
- **`pack_model_h5.py` old-format recon**: looks for
  `recon/patches/recon_{split}_{stem}.tif` (with optional split prefix).
  `split` comes from the `split` column in the primary latents CSV.
- **UMAP pkl transform**: `joblib.load` (not `pickle.load`) — the pkl format
  is protocol 4 and joblib handles it correctly.
